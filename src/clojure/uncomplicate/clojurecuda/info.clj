;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.clojurecuda.info
  "Info functions for all CUDA objects (devices, etc...).
  "
  (:require [clojure.string :as str]
            [uncomplicate.commons.core :refer [with-release Info]]
            [uncomplicate.fluokitten.core :refer [fmap op]]
            [uncomplicate.clojure-cpp :as cpp
             :refer [int-pointer byte-pointer size-t-pointer get-string get-entry]]
            [uncomplicate.clojurecuda.internal
             [constants :refer [ctx-limits dec-compute-mode dec-func-cache-config dec-shared-config
                                dec-stream-flag func-cache-config shared-config-map]]
             [utils :refer [with-check maybe]]
             [impl :refer [current-context* ->CUDevice]]])
  (:import [org.bytedeco.cuda.global cudart]
           [org.bytedeco.cuda.cudart CUctx_st CUfunc_st CUstream_st]
           [uncomplicate.clojurecuda.internal.impl CUDevice]))

;; =================== Info* utility macros ===============================

(defmacro ^:private info-attribute* [method object attribute]
  `(long (with-release [res# (int-pointer 1)]
           (with-check (~method res# ~attribute ~object)
             (get-entry res# 0)))))

;; =================== Version Management =================================

(defn driver-version ^long []
  (with-release [res (int-pointer 1)]
    (with-check (cudart/cuDriverGetVersion res) (get-entry res 0))))

;; =================== Device info  =======================================

(defn device-name [^CUDevice device]
  (with-release [res (byte-pointer 64)]
    (with-check (cudart/cuDeviceGetName res 64 (.dev device))
      (clojure.string/replace (get-string res) #" " ""))))

(defn total-mem [^CUDevice device]
  (with-release [res (size-t-pointer 1)]
    (with-check (cudart/cuDeviceTotalMem res (.dev device))
      (get-entry res 0))))

(defn async-engine-count ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT))

(defn can-map-host-memory [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY)))

(defn clock-rate ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_CLOCK_RATE))

(defn compute-capability-major ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR))

(defn compute-capability-minor ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))

(defn compute-mode [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_COMPUTE_MODE))

(defn concurrent-kernels ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS))

(defn ecc-enabled [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_ECC_ENABLED)))

(defn global-L1-cache-supported [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED)))

(defn global-memory-bus-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH))

(defn integrated [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_INTEGRATED)))

(defn kernel-exec-timeout [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)))

(defn L2-cache-size ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE))

(defn local-L1-cache-supported [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED)))

(defn managed-memory [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)))

(defn concurrent-managed-access [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)))

(defn max-block-dim-x ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X))

(defn max-block-dim-y ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y))

(defn max-block-dim-z ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))

(defn max-grid-dim-x ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X))

(defn max-grid-dim-y ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y))

(defn max-grid-dim-z ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))

(defn max-pitch ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_PITCH))

(defn max-registers-per-block ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK))

(defn max-registers-per-multiprocessor ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR))

(defn max-shared-memory-per-block ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))

(defn max-shared-memory-per-multiprocessor ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))

(defn max-threads-per-block ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))

(defn max-threads-per-multiprocessor ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR))

(defn maximum-surface1d-layered-layers ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS))

(defn maximum-surface1d-layered-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH))

(defn maximum-surface1d-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH))

(defn maximum-surface2d-height ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT))

(defn maximum-surface2d-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH))

(defn maximum-surface2d-layered-height ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT))

(defn maximum-surface2d-layered-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH))

(defn maximum-surface2d-layered-layers ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS))

(defn maximum-surface3d-depth ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH))

(defn maximum-surface3d-height ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT))

(defn maximum-surface3d-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH))

(defn maximum-surfacecubemap-layered-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH))

(defn maximum-surfacecubemap-layered-layers ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS))

(defn maximum-surfacecubemap-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH))

(defn maximum-texture1d-layered-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH))

(defn maximum-texture1d-layered-layers ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS))

(defn maximum-texture1d-linear-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH))

(defn maximum-texture1d-mipmapped-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH))

(defn maximum-texture1d-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH))

(defn maximum-texture2d-height ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT))

(defn maximum-texture2d-layered-height ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT))

(defn maximum-texture2d-layered-layers ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS))

(defn maximum-texture2d-linear-height ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT))

(defn maximum-texture2d-linear-pitch ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH))

(defn maximum-texture2d-linear-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH))

(defn maximum-texture2d-mipmapped-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH))

(defn maximum-texture2d-mipmapped-height ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT))

(defn maximum-texture2d-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH))

(defn maximum-texture3d-depth ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH))

(defn maximum-texture3d-depth-alternate ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE))

(defn maximum-texture3d-height ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT))

(defn maximum-texture3d-height-alternate ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE))

(defn maximum-texture3d-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH))

(defn maximum-texture3d-width-alternate ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE))

(defn maximum-texturecubemap-layered-layers ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS))

(defn maximum-texturecubemap-layered-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH))

(defn maximum-texturecubemap-width ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH))

(defn memory-clock-rate ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE))

(defn multi-gpu-board [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD)))

(defn multi-gpu-board-group-id ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID))

(defn multiprocessor-count ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))

(defn pci-bus-id ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_PCI_BUS_ID))

(defn pci-bus-id-string [^CUDevice device]
  (with-release [res (byte-pointer 13)
                 res2 (byte-pointer 12)]
    (with-check (cudart/cuDeviceGetPCIBusId res 13 (.dev device))
      (do
        (cpp/memcpy! res res2 12)
        (get-string res2)))))

(defn pci-device-id ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID))

(defn pci-domain-id ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID))

(defn stream-priorities-supported [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED)))

(defn surface-alignment ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT))

(defn tcc-driver [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_TCC_DRIVER)))

(defn texture-alignment ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT))

(defn texture-pitch-alignment ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT))

(defn total-constant-memory ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY))

(defn unified-addressing [^CUDevice device]
  (pos? (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                         cudart/CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)))

(defn warp-size ^long [^CUDevice device]
  (info-attribute* cudart/cuDeviceGetAttribute (.dev device)
                   cudart/CU_DEVICE_ATTRIBUTE_WARP_SIZE))

(def ^:no-doc
  device-attributes
  {:name device-name
   :total-mem total-mem
   :async-engine-count async-engine-count
   :can-map-host-memory can-map-host-memory
   :clock-rate clock-rate
   :compute-capability-major compute-capability-major
   :compute-capability-minor compute-capability-minor
   :compute-mode (comp dec-compute-mode compute-mode)
   :concurrent-kernels concurrent-kernels
   :ecc-enabled ecc-enabled
   :global-L1-cache-supported global-L1-cache-supported
   :global-memory-bus-width global-memory-bus-width
   :integrated integrated
   :kernel-exec-timeout kernel-exec-timeout
   :L2-cache-size L2-cache-size
   :local-L1-cache-supported local-L1-cache-supported
   :managed-memory managed-memory
   :max-block-dim-x max-block-dim-x
   :max-block-dim-y max-block-dim-y
   :max-block-dim-z max-block-dim-z
   :max-grid-dim-x max-grid-dim-x
   :max-grid-dim-y max-grid-dim-y
   :max-grid-dim-z max-grid-dim-z
   :max-pitch max-pitch
   :max-registers-per-block max-registers-per-block
   :max-registers-per-multiprocessor max-registers-per-multiprocessor
   :max-shared-memory-per-block max-shared-memory-per-block
   :max-shared-memory-per-multiprocessor max-shared-memory-per-multiprocessor
   :max-threads-per-block max-threads-per-block
   :max-threads-per-multiprocessor max-threads-per-multiprocessor
   :maximum-surface1d-layered-layers maximum-surface1d-layered-layers
   :maximum-surface1d-layered-width maximum-surface1d-layered-width
   :maximum-surface1d-width maximum-surface1d-width
   :maximum-surface2d-height maximum-surface2d-height
   :maximum-surface2d-width maximum-surface2d-width
   :maximum-surface2d-layered-height maximum-surface2d-layered-height
   :maximum-surface2d-layered-width maximum-surface2d-layered-width
   :maximum-surface2d-layered-layers maximum-surface2d-layered-layers
   :maximum-surface3d-depth maximum-surface3d-depth
   :maximum-surface3d-height maximum-surface3d-height
   :maximum-surface3d-width maximum-surface3d-width
   :maximum-surfacecubemap-layered-width maximum-surfacecubemap-layered-width
   :maximum-surfacecubemap-layered-layers maximum-surfacecubemap-layered-layers
   :maximum-surfacecubemap-width maximum-surfacecubemap-width
   :maximum-texture1d-layered-width maximum-texture1d-layered-width
   :maximum-texture1d-layered-layers maximum-texture1d-layered-layers
   :maximum-texture1d-linear-width maximum-texture1d-linear-width
   :maximum-texture1d-mipmapped-width maximum-texture1d-mipmapped-width
   :maximum-texture1d-width maximum-texture1d-width
   :maximum-texture2d-height maximum-texture2d-height
   :maximum-texture2d-layered-height maximum-texture2d-layered-height
   :maximum-texture2d-layered-layers maximum-texture2d-layered-layers
   :maximum-texture2d-linear-height maximum-texture2d-linear-height
   :maximum-texture2d-linear-pitch maximum-texture2d-linear-pitch
   :maximum-texture2d-linear-width maximum-texture2d-linear-width
   :maximum-texture2d-mipmapped-width maximum-texture2d-mipmapped-width
   :maximum-texture2d-mipmapped-height maximum-texture2d-mipmapped-height
   :maximum-texture2d-width maximum-texture2d-width
   :maximum-texture3d-depth maximum-texture3d-depth
   :maximum-texture3d-depth-alternate maximum-texture3d-depth-alternate
   :maximum-texture3d-height maximum-texture3d-height
   :maximum-texture3d-height-alternate maximum-texture3d-height-alternate
   :maximum-texture3d-width maximum-texture3d-width
   :maximum-texture3d-width-alternate maximum-texture3d-width-alternate
   :maximum-texturecubemap-layered-layers maximum-texturecubemap-layered-layers
   :maximum-texturecubemap-layered-width maximum-texturecubemap-layered-width
   :maximum-texturecubemap-width maximum-texturecubemap-width
   :memory-clock-rate memory-clock-rate
   :multi-gpu-board multi-gpu-board
   :multi-gpu-board-group-id multi-gpu-board-group-id
   :multiprocessor-count multiprocessor-count
   :pci-bus-id pci-bus-id
   :pci-bus-id-string pci-bus-id-string
   :pci-device-id pci-device-id
   :pci-domain-id pci-domain-id
   :stream-priorities-supported stream-priorities-supported
   :surface-alignment surface-alignment
   :tcc-driver tcc-driver
   :texture-alignment texture-alignment
   :texture-pitch-alignment texture-pitch-alignment
   :total-constant-memory total-constant-memory
   :unified-addressing unified-addressing
   :warp-size warp-size})

(extend-type CUDevice
  Info
  (info
    ([d attribute]
     (if-let [attribute-fn (device-attributes attribute)]
       (maybe (attribute-fn d))
       (throw (ex-info "Unknown attribute." {:attribute attribute}))))
    ([d]
     (fmap #(maybe (% d)) device-attributes))))

;; =======================  Context Info ==================================

(defn api-version
  "Gets the context's API version."
  ([^CUctx_st ctx]
   (with-release [res (int-pointer 1)]
     (with-check (cudart/cuCtxGetApiVersion ctx res) (get-entry res 0))))
  ([]
   (with-release [res (int-pointer 1)]
     (with-check (cudart/cuCtxGetApiVersion ^CUctx_st (current-context*) res)
       (get-entry res 0)))))

(defn cache-config
  "Returns the preferred cache configuration for the current context.

  See [cuCtxGetCacheConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  []
  (with-release [res (int-pointer 1)]
    (dec-func-cache-config (with-check (cudart/cuCtxGetCacheConfig res) (get-entry res 0)))))

(defn limit*
  "Returns or sets resource limits for the attribute specified by integer `limit`.

  See [cuCtxGetLimit](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  (^long [limit]
   (with-release [res (size-t-pointer 1)]
     (with-check (cudart/cuCtxGetLimit res limit) (get-entry res 0))))
  (^long [limit ^long value]
   (with-check (cudart/cuCtxSetLimit limit value) value)))

(defn limit
  "Returns resource limits for the attribute specified by keyword `limit`.

  Supported limits are: `stack-size`, `malloc-heap-size`, `printf-fifo-size`, `dev-runtime-sync-depth`,
  `dev-runtime-pending-launch-count`.

  See [cuCtxGetLimit](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  ^long [limit]
  (limit* (or (ctx-limits limit) (throw (ex-info "Unknown limit." {:limit limit :available ctx-limits})))))

(defn limit!
  "Sets resource limit for the attribute specified by keyword `limit` to `value`.

  Supported limits are: `stack-size`, `malloc-heap-size`, `printf-fifo-size`, `dev-runtime-sync-depth`,
  `dev-runtime-pending-launch-count`.

  See [cuCtxGetLimit](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  ^long [limit ^long value]
  (limit* (or (ctx-limits limit) (throw (ex-info "Unknown limit." {:limit limit :available ctx-limits})))
          value))

(defn ctx-device
  "Returns the device for the current context."
  []
  (with-release [res (int-pointer 1)]
    (with-check (cudart/cuCtxGetDevice res) (->CUDevice (get-entry res 0)))))

(defn shared-config*
  "Sets or gets the current shared memory configuration for the current context or kernel `func`.

  See [cuCtxGetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  See [cuCtxSetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  See [cuFuncSetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
  "
  (^long []
   (with-release [res (int-pointer 1)]
     (with-check (cudart/cuCtxGetSharedMemConfig res) (get-entry res 0))))
  (^long [^long config]
   (with-check (cudart/cuCtxSetSharedMemConfig config) config))
  ([^CUfunc_st func ^long config]
   (with-check (cudart/cuFuncSetSharedMemConfig func config) func)))

(defn shared-config
  "Gets the current shared memory configuration for the current context.

  See [cuCtxGetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  []
  (dec-shared-config (shared-config*)))

(defn shared-config!
  "Sets the current shared memory configuration for the current context.

  See [cuCtxSetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  See [cuFuncSetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
  "
  ([config]
   (shared-config* (or (shared-config-map config)
                       (ex-info "Unknown config." {:config config :available shared-config-map}))))
  ([func config]
   (shared-config* func (or (shared-config-map config)
                            (ex-info "Unknown config." {:config config :available shared-config-map})))))

(defn stream-priority-range
  "Returns a vector of 2 numerical values that correspond to the least and greatest stream priorities.

  See [cuCtxGetStreamPriorityRange](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  []
  (with-release [least (int-pointer 1)
                 greatest (int-pointer 1)]
    (with-check (cudart/cuCtxGetStreamPriorityRange least greatest)
      [(get-entry least 0) (get-entry greatest 0)])))

(extend-type CUctx_st
  Info
  (info
    ([_ info-type]
     (maybe
      (case info-type
        :api-version (api-version)
        :cache-config (cache-config)
        :stack-size (limit* cudart/CU_LIMIT_STACK_SIZE)
        :malloc-heap-size (limit* cudart/CU_LIMIT_MALLOC_HEAP_SIZE)
        :printf-fifo-size (limit* cudart/CU_LIMIT_PRINTF_FIFO_SIZE)
        :dev-runtime-sync-depth (limit* cudart/CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH)
        :dev-runtime-pending-launch-count (limit* cudart/CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT)
        :limits (fmap #(maybe (limit* %)) ctx-limits)
        :device (ctx-device)
        :shared-config (shared-config)
        :stream-priority-range (stream-priority-range)
        nil)))
    ([_]
     (op {:api-version (maybe (api-version))
          :cache-config (maybe (cache-config))
          :device (maybe (ctx-device))
          :shared-config (shared-config)
          :stream-priority-range (stream-priority-range)}
         (fmap #(maybe (limit* %)) ctx-limits)))))

;; =========================== Stream Management ================================

(defn stream-flag [^CUstream_st hstream]
  (with-release [res (int-pointer 1)]
    (with-check (cudart/cuStreamGetFlags hstream res) (get-entry res 0))))

(defn stream-priority ^long [^CUstream_st hstream]
  (with-release [res (int-pointer 1)]
    (with-check (cudart/cuStreamGetPriority hstream res) (get-entry res 0))))

(extend-type CUstream_st
  Info
  (info
    ([hstream info-type]
     (maybe
      (case info-type
        :flag (dec-stream-flag (stream-flag hstream))
        :priority (stream-priority hstream)
        nil)))
    ([hstream]
     {:flag (maybe (dec-stream-flag (stream-flag hstream)))
      :priority (maybe (stream-priority hstream))})))

;; ============================= Execution Management ==========================

(defn max-threads-per-block-fn ^long [^CUfunc_st function]
  (info-attribute* cudart/cuFuncGetAttribute function
                   cudart/CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK))

(defn shared-size ^long [^CUfunc_st function]
  (info-attribute* cudart/cuFuncGetAttribute function
                   cudart/CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES))

(defn const-size ^long [^CUfunc_st function]
  (info-attribute* cudart/cuFuncGetAttribute function
                   cudart/CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES))

(defn local-size ^long [^CUfunc_st function]
  (info-attribute* cudart/cuFuncGetAttribute function
                   cudart/CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES))

(defn num-regs ^long [^CUfunc_st function]
  (info-attribute* cudart/cuFuncGetAttribute function
                   cudart/CU_FUNC_ATTRIBUTE_NUM_REGS))

(defn ptx-version ^long [^CUfunc_st function]
  (info-attribute* cudart/cuFuncGetAttribute function
                   cudart/CU_FUNC_ATTRIBUTE_PTX_VERSION))

(defn binary-version ^long [^CUfunc_st function]
  (info-attribute* cudart/cuFuncGetAttribute function
                   cudart/CU_FUNC_ATTRIBUTE_BINARY_VERSION))

(defn cache-config*
  "Sets the preferred cache configuration for a device function `fun`, as an integer `config`.

  See [cuFuncSetCacheConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
  "
  [fun ^long config]
  (with-check (cudart/cuFuncSetCacheConfig fun config) fun))

(defn cache-config!
  "Sets the preferred cache configuration for a device function `fun`, as a keyword `config`.

  Available configs are `:prefer-none`, `:prefer-shared`, `:prefer-L1`, and `:prefer-equal`.

  See [cuFuncSetCacheConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
  "
  [fun config]
  (cache-config* fun (or (func-cache-config config)
                         (throw (ex-info "Invaling cache config."
                                         {:config config :available func-cache-config})))))

(extend-type CUfunc_st
  Info
  (info
    ([fun info-type]
     (maybe
      (case info-type
        :max-threads-per-block (max-threads-per-block-fn fun)
        :shared-size (shared-size fun)
        :const-size (const-size fun)
        :local-size (local-size fun)
        :num-regs (num-regs fun)
        :ptx-version (ptx-version fun)
        :binary-version (binary-version fun)
        nil)))
    ([fun]
     {:max-threads-per-block (maybe (max-threads-per-block-fn fun))
      :shared-size (maybe (shared-size fun))
      :const-size (maybe (const-size fun))
      :local-size (maybe (local-size fun))
      :num-regs (maybe (num-regs fun))
      :ptx-version (maybe (ptx-version fun))
      :binary-version (maybe (binary-version fun))})))
