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
  (:import [jcuda.driver CUresult CUctx_flags JCudaDriver CUdevice_attribute CUcomputemode
            CUmemAttach_flags]))

;; ============= Error Codes ===================================================

(defn dec-error
  "Decodes CUDA error code to a meaningful string."
  [^long code]
  (CUresult/stringFor code))

;; ==================== Keyword mapping ======================================

(def ^{:doc "Available context flags defined in the CUDA standard.
See [cuCtxCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf)."}
  ctx-flags
  {:sched-auto CUctx_flags/CU_CTX_SCHED_AUTO
   :sched-spin CUctx_flags/CU_CTX_SCHED_SPIN
   :sched-yield CUctx_flags/CU_CTX_SCHED_YIELD
   :sched-blocking-sync CUctx_flags/CU_CTX_SCHED_BLOCKING_SYNC
   :map-host CUctx_flags/CU_CTX_MAP_HOST
   :lmem-resize-to-max CUctx_flags/CU_CTX_LMEM_RESIZE_TO_MAX})

(def ^{:doc "Available flags for the [[core/mem-host-alloc]] function.
See [cuCtxMemHostAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9)."}
  mem-host-alloc-flags
  {:portable JCudaDriver/CU_MEMHOSTALLOC_PORTABLE
   :devicemap JCudaDriver/CU_MEMHOSTALLOC_DEVICEMAP
   :writecombined JCudaDriver/CU_MEMHOSTALLOC_WRITECOMBINED})

(def ^{:doc "Available flags for the [[core/mem-host-attach]] function.
See [cuMemAllocManaged](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32)."}
  mem-attach-flags
  {:global CUmemAttach_flags/CU_MEM_ATTACH_GLOBAL
   :host CUmemAttach_flags/CU_MEM_ATTACH_HOST})

(defn dec-compute-mode [^long mode]
  (case mode
    0 :default
    1 :exclusive
    2 :prohibited
    3 :exclusive-process
    mode) )
