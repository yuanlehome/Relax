// Copyright (c) 2026 Relax Authors. All Rights Reserved.

'use strict'

// Older Node tools size worker pools from os.cpus(), which ignores CPU affinity.
// This changes only their CPU-count heuristic; cgroups still enforce resources.
const os = require('os')
const fs = require('fs')
const originalCpus = os.cpus
const availableParallelism = os.availableParallelism

function visibleCpuCount() {
  try {
    if (typeof availableParallelism === 'function') {
      const count = availableParallelism()
      if (Number.isSafeInteger(count) && count > 0) return count
    }
    const match = /^Cpus_allowed_list:\s*([^\n]+)$/m.exec(
      fs.readFileSync('/proc/self/status', 'utf8')
    )
    if (!match) return null
    let count = 0
    let previous = -1
    for (const item of match[1].trim().split(',')) {
      const range = /^(\d+)(?:-(\d+))?$/.exec(item)
      if (!range) return null
      const first = Number(range[1])
      const last = Number(range[2] === undefined ? range[1] : range[2])
      if (!Number.isSafeInteger(last) || first <= previous || last < first) return null
      count += last - first + 1
      previous = last
    }
    return Number.isSafeInteger(count) && count > 0 ? count : null
  } catch (_) {
    return null
  }
}

try {
  os.cpus = function cpus() {
    const cpus = originalCpus()
    const count = visibleCpuCount()
    return Array.isArray(cpus) && count !== null && count < cpus.length
      ? cpus.slice(0, count)
      : cpus
  }
  const syncBuiltinESMExports = require('module').syncBuiltinESMExports
  if (typeof syncBuiltinESMExports === 'function') syncBuiltinESMExports()
} catch (_) {
  try {
    os.cpus = originalCpus
  } catch (_) {
    // A previously frozen builtin must not turn this optional preload into a startup error.
  }
}
