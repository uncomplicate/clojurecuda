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
  (:import [jcuda.driver CUresult CUctx_flags JCudaDriver CUdevice_attribute CUcomputemode]))

;; ============= Error Codes ===================================================

(defn dec-error
  "Decodes CUDA error code to a meaningful string."
  [^long code]
  (CUresult/stringFor code))

;; ==================== Keyword mapping ======================================

(def ctx-flags
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



(def device-attributes-special
  {:compute-mode CUdevice_attribute/CU_DEVICE_ATTRIBUTE_COMPUTE_MODE})

(def device-attributes-bool
  {:async-engine-count CUdevice_attribute/CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
   :can-map-host-memory CUdevice_attribute/CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
   :clock-rate CUdevice_attribute/CU_DEVICE_ATTRIBUTE_CLOCK_RATE
   :compute-capability-major CUdevice_attribute/CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
   :compute-capability-minor CUdevice_attribute/CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
   :compute-mode CUdevice_attribute/CU_DEVICE_ATTRIBUTE_COMPUTE_MODE
   :concxrrent-kernels CUdevice_attribute/CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS
   :ecc-enabled CUdevice_attribute/CU_DEVICE_ATTRIBUTE_ECC_ENABLED
   :global-L1-cache-supported CUdevice_attribute/CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED
   :global-memory-bus-widht CUdevice_attribute/CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
   :integrated CUdevice_attribute/CU_DEVICE_ATTRIBUTE_INTEGRATED
   :kernel-exec-timeout CUdevice_attribute/CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT
   :L2-cache-size CUdevice_attribute/CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
   :local-L1-cache-supported CUdevice_attribute/CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED
   :managed-memory CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY


   })

(defn dec-compute-mode [^long mode]
  (case mode
    0 :default
    1 :exclusive
    2 :prohibited
    3 :exclusive-process
    mode) )
