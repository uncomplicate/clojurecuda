;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.clojurecuda.constants
  "Defines constants and mappings from/to CUDA constants."
  (:import [jcuda.driver CUresult CUctx_flags JCudaDriver]))

;; ============= Error Codes ===================================================

(defn dec-error
  "Decodes CUDA error code to a meaningful string."
  [^long code]
  (CUresult/stringFor code))

;; ==================== Keyword mapping ======================================

(def ^{:doc "Types of OpenCL devices defined in OpenCL standard.
See http://www.khronos.org/registry/cl/sdk/2.0/docs/man/xhtml/clGetDeviceIDs.html"}
  ctx-flags
  {:sched-auto CUctx_flags/CU_CTX_SCHED_AUTO
   :sched-spin CUctx_flags/CU_CTX_SCHED_SPIN
   :sched-yield CUctx_flags/CU_CTX_SCHED_YIELD
   :block-sync CUctx_flags/CU_CTX_BLOCKING_SYNC
   :sched-blocking-sync CUctx_flags/CU_CTX_SCHED_BLOCKING_SYNC
   :sched-mask CUctx_flags/CU_CTX_SCHED_MASK
   :lmem-resize-to-max CUctx_flags/CU_CTX_LMEM_RESIZE_TO_MAX
   :flags-mask CUctx_flags/CU_CTX_FLAGS_MASK})

(def mem-host-alloc-flags
  {:portable JCudaDriver/CU_MEMHOSTALLOC_PORTABLE
   :device-map JCudaDriver/CU_MEMHOSTALLOC_DEVICEMAP
   :write-combined JCudaDriver/CU_MEMHOSTALLOC_WRITECOMBINED})
