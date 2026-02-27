import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    // Use threads instead of forks to avoid worker termination timeouts
    pool: 'threads',
    // Increase teardown timeout for file-handle cleanup
    teardownTimeout: 10000,
    // Exclude benchmark from default test run (run separately: npx vitest run rangeImageBenchmark)
    exclude: ['**/rangeImageBenchmark*', '**/node_modules/**'],
  },
})
