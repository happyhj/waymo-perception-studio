/**
 * Browser port of Waymo Open Dataset v2's merge() function.
 *
 * Original Python: waymo_open_dataset/v2/dataframe_utils.py
 * https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/dataframe_utils.py
 *
 * Key idea: all Waymo v2 Parquet tables share `key.*` columns
 * (e.g. key.segment_context_name, key.frame_timestamp_micros, key.camera_name).
 * This module auto-detects common keys and joins rows via Map lookup.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A generic row from a parsed Parquet table — flat key/value object. */
export type ParquetRow = Record<string, unknown>

export interface MergeOptions {
  /** Group left-side rows by common keys before join (agg into arrays). */
  leftGroup?: boolean
  /** Group right-side rows by common keys before join (agg into arrays). */
  rightGroup?: boolean
  /** Key column prefix. Default: 'key.' */
  keyPrefix?: string
}

// ---------------------------------------------------------------------------
// Internal helpers (mirror the Python SDK helpers)
// ---------------------------------------------------------------------------

/** Detect key columns by prefix. */
function selectKeyColumns(row: ParquetRow, prefix: string): Set<string> {
  return new Set(Object.keys(row).filter((c) => c.startsWith(prefix)))
}

/** Build a composite lookup key from a row. */
function compositeKey(row: ParquetRow, keys: string[]): string {
  return keys.map((k) => String(row[k])).join('|')
}

/**
 * Group rows by `groupKeys`, aggregating all other columns into arrays.
 * Equivalent to Python's `df.groupby(keys).agg(list).reset_index()`.
 */
function groupBy(rows: ParquetRow[], groupKeys: string[]): ParquetRow[] {
  const map = new Map<string, ParquetRow>()

  for (const row of rows) {
    const key = compositeKey(row, groupKeys)
    if (!map.has(key)) {
      // Seed with key columns as scalars, everything else as single-element arrays
      const seed: ParquetRow = {}
      for (const k of groupKeys) seed[k] = row[k]
      for (const col of Object.keys(row)) {
        if (!groupKeys.includes(col)) seed[col] = [row[col]]
      }
      map.set(key, seed)
    } else {
      const existing = map.get(key)!
      for (const col of Object.keys(row)) {
        if (!groupKeys.includes(col)) {
          ;(existing[col] as unknown[]).push(row[col])
        }
      }
    }
  }

  return Array.from(map.values())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Merge two Parquet-parsed row arrays on their common `key.*` columns.
 *
 * Mirrors `waymo_open_dataset.v2.merge()`:
 * 1. Auto-detect key columns via prefix.
 * 2. Find common keys (intersection).
 * 3. Optionally group by common keys to avoid cartesian products.
 * 4. Inner join via Map lookup.
 *
 * @example
 * ```ts
 * // Join vehicle_pose (199 rows) with lidar_box (18633 rows)
 * const merged = merge(poseRows, boxRows)
 * // → 18633 rows, each box row enriched with its frame's pose
 *
 * // Join with grouping to get all boxes per frame as arrays
 * const grouped = merge(poseRows, boxRows, { rightGroup: true })
 * // → 199 rows, each pose row has boxes[] array
 * ```
 */
export function merge(
  left: ParquetRow[],
  right: ParquetRow[],
  options: MergeOptions = {},
): ParquetRow[] {
  const { leftGroup = false, rightGroup = false, keyPrefix = 'key.' } = options

  if (left.length === 0 || right.length === 0) return []

  // 1. Detect key columns
  const leftKeys = selectKeyColumns(left[0], keyPrefix)
  const rightKeys = selectKeyColumns(right[0], keyPrefix)

  // 2. Common keys (intersection)
  const commonKeys = [...leftKeys].filter((k) => rightKeys.has(k))
  if (commonKeys.length === 0) {
    console.warn('[merge] No common key columns found. Returning empty.')
    return []
  }
  // Sort for deterministic composite key
  commonKeys.sort()

  // 3. Optional grouping
  let leftRows = left
  let rightRows = right
  if (leftGroup && leftKeys.size !== commonKeys.length) {
    leftRows = groupBy(left, commonKeys)
  }
  if (rightGroup && rightKeys.size !== commonKeys.length) {
    rightRows = groupBy(right, commonKeys)
  }

  // 4. Build lookup from the smaller side for perf
  const rightMap = new Map<string, ParquetRow[]>()
  for (const row of rightRows) {
    const key = compositeKey(row, commonKeys)
    if (!rightMap.has(key)) rightMap.set(key, [])
    rightMap.get(key)!.push(row)
  }

  // 5. Inner join
  const result: ParquetRow[] = []
  for (const lRow of leftRows) {
    const key = compositeKey(lRow, commonKeys)
    const matches = rightMap.get(key)
    if (!matches) continue
    for (const rRow of matches) {
      result.push({ ...lRow, ...rRow })
    }
  }

  return result
}

// ---------------------------------------------------------------------------
// Convenience: index rows by a single key for O(1) lookup
// ---------------------------------------------------------------------------

/**
 * Build a Map from rows indexed by a single column value.
 * Useful for quick frame lookup: `indexBy(poseRows, 'key.frame_timestamp_micros')`
 */
export function indexBy<T extends ParquetRow>(
  rows: T[],
  column: string,
): Map<unknown, T> {
  const map = new Map<unknown, T>()
  for (const row of rows) {
    map.set(row[column], row)
  }
  return map
}

/**
 * Build a Map from rows grouped by a column value.
 * Returns arrays for 1:N relationships.
 * e.g. `groupIndexBy(boxRows, 'key.frame_timestamp_micros')` → Map<timestamp, Box[]>
 */
export function groupIndexBy<T extends ParquetRow>(
  rows: T[],
  column: string,
): Map<unknown, T[]> {
  const map = new Map<unknown, T[]>()
  for (const row of rows) {
    const key = row[column]
    if (!map.has(key)) map.set(key, [])
    map.get(key)!.push(row)
  }
  return map
}
