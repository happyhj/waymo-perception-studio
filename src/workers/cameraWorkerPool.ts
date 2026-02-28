/**
 * Camera Worker Pool — manages N workers for parallel camera image loading.
 *
 * Same architecture as WorkerPool (lidar) but for camera_image Parquet.
 * Camera images don't need CPU conversion (just BROTLI decompress + JPEG pass-through),
 * so the pool mainly parallelizes Parquet decompression.
 */

import type {
  CameraWorkerRequest,
  CameraWorkerResponse,
  CameraWorkerRowGroupResult,
  CameraWorkerReady,
} from './cameraWorker'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CameraPoolInitOptions {
  cameraUrl: string
}

interface PendingRequest {
  resolve: (result: CameraWorkerRowGroupResult) => void
  reject: (err: Error) => void
}

interface PoolWorker {
  worker: Worker
  busy: boolean
  ready: boolean
}

// ---------------------------------------------------------------------------
// CameraWorkerPool
// ---------------------------------------------------------------------------

export class CameraWorkerPool {
  private workers: PoolWorker[] = []
  private pendingRequests = new Map<number, PendingRequest>()
  private nextRequestId = 0
  private numRowGroups = 0
  private waitQueue: Array<{
    requestId: number
    rowGroupIndex: number
    resolve: (result: CameraWorkerRowGroupResult) => void
    reject: (err: Error) => void
  }> = []

  readonly concurrency: number

  constructor(concurrency: number) {
    this.concurrency = concurrency
  }

  async init(opts: CameraPoolInitOptions): Promise<{ numRowGroups: number }> {
    const readyPromises: Promise<CameraWorkerReady>[] = []

    for (let i = 0; i < this.concurrency; i++) {
      const worker = new Worker(
        new URL('./cameraWorker.ts', import.meta.url),
        { type: 'module' },
      )

      const poolWorker: PoolWorker = { worker, busy: false, ready: false }
      this.workers.push(poolWorker)

      const readyPromise = new Promise<CameraWorkerReady>((resolve, reject) => {
        worker.onmessage = (e: MessageEvent<CameraWorkerResponse>) => {
          if (e.data.type === 'ready') {
            poolWorker.ready = true
            worker.onmessage = (ev: MessageEvent<CameraWorkerResponse>) =>
              this.handleWorkerMessage(i, ev)
            resolve(e.data)
          } else if (e.data.type === 'error') {
            reject(new Error(e.data.message))
          }
        }
        worker.onerror = (e) => reject(new Error(e.message))
      })

      readyPromises.push(readyPromise)

      worker.postMessage({
        type: 'init',
        cameraUrl: opts.cameraUrl,
      } satisfies CameraWorkerRequest)
    }

    const results = await Promise.all(readyPromises)
    this.numRowGroups = results[0].numRowGroups
    return { numRowGroups: this.numRowGroups }
  }

  getNumRowGroups(): number {
    return this.numRowGroups
  }

  isReady(): boolean {
    return this.workers.some((w) => w.ready)
  }

  requestRowGroup(rowGroupIndex: number): Promise<CameraWorkerRowGroupResult> {
    return new Promise((resolve, reject) => {
      const requestId = this.nextRequestId++
      const idle = this.workers.find((w) => w.ready && !w.busy)
      if (idle) {
        this.dispatchToWorker(idle, requestId, rowGroupIndex, resolve, reject)
      } else {
        this.waitQueue.push({ requestId, rowGroupIndex, resolve, reject })
      }
    })
  }

  terminate(): void {
    for (const pw of this.workers) {
      pw.worker.terminate()
    }
    this.workers = []
    this.pendingRequests.clear()
    this.waitQueue = []
  }

  // -------------------------------------------------------------------------
  // Internal
  // -------------------------------------------------------------------------

  private dispatchToWorker(
    pw: PoolWorker,
    requestId: number,
    rowGroupIndex: number,
    resolve: (result: CameraWorkerRowGroupResult) => void,
    reject: (err: Error) => void,
  ): void {
    pw.busy = true
    this.pendingRequests.set(requestId, { resolve, reject })
    pw.worker.postMessage({
      type: 'loadRowGroup',
      requestId,
      rowGroupIndex,
    } satisfies CameraWorkerRequest)
  }

  private handleWorkerMessage(
    workerIndex: number,
    e: MessageEvent<CameraWorkerResponse>,
  ): void {
    const msg = e.data
    const pw = this.workers[workerIndex]

    if (msg.type === 'rowGroupReady' || msg.type === 'error') {
      const rid = 'requestId' in msg ? msg.requestId : -1
      const pending = this.pendingRequests.get(rid ?? -1)
      if (pending) {
        this.pendingRequests.delete(rid!)
        if (msg.type === 'error') {
          pending.reject(new Error(msg.message))
        } else {
          pending.resolve(msg)
        }
      }

      pw.busy = false
      this.drainQueue()
    }
  }

  private drainQueue(): void {
    while (this.waitQueue.length > 0) {
      const idle = this.workers.find((w) => w.ready && !w.busy)
      if (!idle) break
      const next = this.waitQueue.shift()!
      this.dispatchToWorker(idle, next.requestId, next.rowGroupIndex, next.resolve, next.reject)
    }
  }
}
