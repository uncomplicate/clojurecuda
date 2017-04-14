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
  (:import [jcuda.driver CUctx_flags JCudaDriver CUdevice_attribute CUcomputemode CUmemAttach_flags
            CUfunc_cache CUdevice_P2PAttribute CUlimit CUsharedconfig]))

;; ==================== Keyword mapping ======================================

(def ^{:const true
       :doc "Available context flags defined in the CUDA standard.
See [cuCtxCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)."}
  ctx-flags
  {:sched-auto CUctx_flags/CU_CTX_SCHED_AUTO
   :sched-spin CUctx_flags/CU_CTX_SCHED_SPIN
   :sched-yield CUctx_flags/CU_CTX_SCHED_YIELD
   :sched-blocking-sync CUctx_flags/CU_CTX_SCHED_BLOCKING_SYNC
   :map-host CUctx_flags/CU_CTX_MAP_HOST
   :lmem-resize-to-max CUctx_flags/CU_CTX_LMEM_RESIZE_TO_MAX})

(def ^{:const true
       :doc "Available context limits.
See [cuCtxGetLimit](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)."}
  ctx-limits
  {:stack-size CUlimit/CU_LIMIT_STACK_SIZE
   :malloc-heap-size CUlimit/CU_LIMIT_MALLOC_HEAP_SIZE
   :printf-fifo-size CUlimit/CU_LIMIT_PRINTF_FIFO_SIZE
   :dev-runtime-sync-depth CUlimit/CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
   :dev-runtime-pending-launch-count CUlimit/CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT})

(def ^{:const true
       :doc "Available shared memory configurations.
See [cuCtxGetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)."}
  ctx-shared-config
  {:default-bank-size CUsharedconfig/CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE
   :four-byte-bank-size CUsharedconfig/CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE
   :eight-byte-bank-size CUsharedconfig/CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE})

(defn dec-shared-config [^long config]
  (case config
    0 :default-bank-size
    1 :four-byte-bank-size
    2 :eight-byte-bank-size
    config))

(defn dec-func-cache [^long mode]
  (case mode
    3 :prefer-equal
    2 :prefer-L1
    0 :prefer-none
    1 :prefer-shared
    mode))

(def ^{:const true
       :doc "Available device P2P attributes.
See [CUdevice_P2PAttribute](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)."}
  p2p-attributes
  {:performance-rank CUdevice_P2PAttribute/CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK
   :access-supported CUdevice_P2PAttribute/CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED
   :native-atomic-supported CUdevice_P2PAttribute/CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED})

(defn dec-compute-mode [^long mode]
  (case mode
    0 :default
    1 :exclusive
    2 :prohibited
    3 :exclusive-process
    mode) )

(def ^{:const true
       :doc "Available flags for the [[core/mem-host-alloc]] function.
See [cuCtxMemHostAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)."}
  mem-host-alloc-flags
  {:portable JCudaDriver/CU_MEMHOSTALLOC_PORTABLE
   :devicemap JCudaDriver/CU_MEMHOSTALLOC_DEVICEMAP
   :writecombined JCudaDriver/CU_MEMHOSTALLOC_WRITECOMBINED})

(def ^{:const true
       :doc "Available flags for the [[core/mem-host-register]] function.
See [cuMemHostRegister](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)."}
  mem-host-register-flags
  {:portable JCudaDriver/CU_MEMHOSTREGISTER_PORTABLE
   :devicemap JCudaDriver/CU_MEMHOSTREGISTER_DEVICEMAP})

(def ^{:const true
       :doc "Available flags for the [[core/mem-host-attach]] function.
See [cuMemAllocManaged](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)."}
  mem-attach-flags
  {:global CUmemAttach_flags/CU_MEM_ATTACH_GLOBAL
   :host CUmemAttach_flags/CU_MEM_ATTACH_HOST})
