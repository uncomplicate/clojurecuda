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
            [uncomplicate.fluokitten.core :refer [fmap op]]
            [uncomplicate.commons.core :refer [Info info extract wrap extract]]
            [uncomplicate.clojurecuda.internal
             [constants :refer :all]
             [utils :refer [with-check maybe]]
             [impl :refer [current-context*]]])
  (:import jcuda.Pointer
           [jcuda.driver JCudaDriver CUdevice CUdevice_attribute CUcontext CUlimit CUstream
            CUfunction CUfunction_attribute]
           [uncomplicate.clojurecuda.internal.impl CUContext CUStream]))

;; =================== Info* utility macros ===============================

(defmacro ^:private info-string*
  ([method object size]
   `(let [res# (byte-array ~size)
          err# (~method res# ~size ~object)]
      (with-check err#
        (String. res# 0 (int (loop [i# 0]
                               (if (or (= i# ~size) (= (byte 0x00) (aget res# i#)))
                                 i#
                                 (recur (inc i#)))))))))
  ([method object]
   `(info-string* ~method ~object 32)))

(defmacro ^:private info-attribute* [method object attribute]
  `(let [res# (int-array 1)
         err# (~method res# ~attribute ~object)]
     (long (with-check err# (aget res# 0)))))

;; =================== Version Management =================================

(defn driver-version ^long []
  (let [res (int-array 1)]
    (with-check (JCudaDriver/cuDriverGetVersion res) (aget res 0))))

;; =================== Device info  =======================================

(defn device-name [^CUdevice device]
  (info-string* JCudaDriver/cuDeviceGetName device))

(defn total-mem [^CUdevice device]
  (let [res (long-array 1)]
    (with-check (JCudaDriver/cuDeviceTotalMem res device) (aget res 0))))

(defn async-engine-count ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT))

(defn can-map-host-memory [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY)))

(defn clock-rate ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_CLOCK_RATE))

(defn compute-capability-major ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR))

(defn compute-capability-minor ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))

(defn compute-mode [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_COMPUTE_MODE))

(defn concurrent-kernels ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS))

(defn ecc-enabled [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_ECC_ENABLED)))

(defn global-L1-cache-supported [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED)))

(defn global-memory-bus-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH))

(defn integrated [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_INTEGRATED)))

(defn kernel-exec-timeout [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)))

(defn L2-cache-size ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE))

(defn local-L1-cache-supported [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED)))

(defn managed-memory [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)))

(defn max-block-dim-x ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X))

(defn max-block-dim-y ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y))

(defn max-block-dim-z ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))

(defn max-grid-dim-x ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X))

(defn max-grid-dim-y ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y))

(defn max-grid-dim-z ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))

(defn max-pitch ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_PITCH))

(defn max-registers-per-block ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK))

(defn max-registers-per-multiprocessor ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR))

(defn max-shared-memory-per-block ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))

(defn max-shared-memory-per-multiprocessor ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))

(defn max-threads-per-block ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))

(defn max-threads-per-multiprocessor ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR))

(defn maximum-surface1d-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS))

(defn maximum-surface1d-layered-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH))

(defn maximum-surface1d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH))

(defn maximum-surface2d-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT))

(defn maximum-surface2d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH))

(defn maximum-surface2d-layered-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT))

(defn maximum-surface2d-layered-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH))

(defn maximum-surface2d-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS))

(defn maximum-surface3d-depth ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH))

(defn maximum-surface3d-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT))

(defn maximum-surface3d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH))

(defn maximum-surfacecubemap-layered-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH))

(defn maximum-surfacecubemap-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS))

(defn maximum-surfacecubemap-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH))

(defn maximum-texture1d-layered-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH))

(defn maximum-texture1d-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS))

(defn maximum-texture1d-linear-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH))

(defn maximum-texture1d-mipmapped-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH))

(defn maximum-texture1d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH))

(defn maximum-texture2d-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT))

(defn maximum-texture2d-layered-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT))

(defn maximum-texture2d-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS))

(defn maximum-texture2d-linear-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT))

(defn maximum-texture2d-linear-pitch ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH))

(defn maximum-texture2d-linear-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH))

(defn maximum-texture2d-mipmapped-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH))

(defn maximum-texture2d-mipmapped-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT))

(defn maximum-texture2d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH))

(defn maximum-texture3d-depth ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH))

(defn maximum-texture3d-depth-alternate ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE))

(defn maximum-texture3d-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT))

(defn maximum-texture3d-height-alternate ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE))

(defn maximum-texture3d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH))

(defn maximum-texture3d-width-alternate ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE))

(defn maximum-texturecubemap-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS))

(defn maximum-texturecubemap-layered-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH))

(defn maximum-texturecubemap-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH))

(defn memory-clock-rate ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE))

(defn multi-gpu-board [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD)))

(defn multi-gpu-board-group-id ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID))

(defn multiprocessor-count ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))

(defn pci-bus-id ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_PCI_BUS_ID))

(defn pci-bus-id-string [^CUdevice device]
  (let [res (make-array String 1)]
    (with-check (JCudaDriver/cuDeviceGetPCIBusId res 64 device) (aget ^objects res 0))))

(defn pci-device-id ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID))

(defn pci-domain-id ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID))

(defn stream-priorities-supported [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED)))

(defn surface-alignment ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT))

(defn tcc-driver [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_TCC_DRIVER)))

(defn texture-alignment ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT))

(defn texture-pitch-alignment ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT))

(defn total-constant-memory ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY))

(defn unified-addressing [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)))

(defn warp-size ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_WARP_SIZE))

(def device-attributes
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

(extend-type CUdevice
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
  ([ctx]
   (let [res (int-array 1)]
     (with-check (JCudaDriver/cuCtxGetApiVersion (extract ctx) res) (aget res 0))))
  ([]
   (let [res (int-array 1)]
     (with-check (JCudaDriver/cuCtxGetApiVersion (current-context*) res) (aget res 0)))))

(defn cache-config
  "Returns the preferred cache configuration for the current context.

  See [cuCtxGetCacheConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  []
  (let [res (int-array 1)]
    (dec-func-cache-config (with-check (JCudaDriver/cuCtxGetCacheConfig res) (aget res 0)))))

(defn limit*
  "Returns or sets resource limits for the attribute specified by integer `limit`.

  See [cuCtxGetLimit](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  (^long [limit]
   (let [res (long-array 1)]
     (with-check (JCudaDriver/cuCtxGetLimit res limit) (aget res 0))))
  (^long [limit ^long value]
   (with-check (JCudaDriver/cuCtxSetLimit limit value) value)))

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
  "Returns the device ID for the current context."
  []
  (let [res (CUdevice.)]
    (with-check (JCudaDriver/cuCtxGetDevice res) res)))

(defn shared-config*
  "Sets or gets the current shared memory configuration for the current context or kernel `func`.

  See [cuCtxGetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  See [cuCtxSetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  See [cuFuncSetSharedMemConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
  "
  (^long []
   (let [res (int-array 1)]
     (with-check (JCudaDriver/cuCtxGetSharedMemConfig res) (aget res 0))))
  (^long [^long config]
   (with-check (JCudaDriver/cuCtxSetSharedMemConfig config) config))
  ([func ^long config]
   (with-check (JCudaDriver/cuFuncSetSharedMemConfig func config) func)))

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
  (let [least (int-array 1)
        greatest (int-array 1)]
    (with-check (JCudaDriver/cuCtxGetStreamPriorityRange least greatest)
      [(aget least 0) (aget greatest 0)])))

(defn context-info
  "All info of the current context."
  ([info-type]
   (maybe
    (case info-type
      :api-version (api-version)
      :cache-config (cache-config)
      :stack-size (limit* CUlimit/CU_LIMIT_STACK_SIZE)
      :malloc-heap-size (limit* CUlimit/CU_LIMIT_MALLOC_HEAP_SIZE)
      :printf-fifo-size (limit* CUlimit/CU_LIMIT_PRINTF_FIFO_SIZE)
      :dev-runtime-sync-depth (limit* CUlimit/CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH)
      :dev-runtime-pending-launch-count (limit* CUlimit/CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT)
      :limits (fmap limit* ctx-limits)
      :device (ctx-device)
      :shared-config (shared-config)
      :stream-priority-range (stream-priority-range)
      nil)))
  ([]
   (op {:api-version (maybe (api-version))
        :cache-config (maybe (cache-config))
        :device (maybe (ctx-device))
        :shared-config (shared-config)
        :stream-priority-range (stream-priority-range)}
       (maybe (fmap limit* ctx-limits)))))

;;TODO
(extend-type CUContext
  Info
  (info
    ([ctx info-type]
     (maybe
      (case info-type
        :api-version (api-version ctx)
        nil)))
    ([ctx]
     {:api-version (maybe (api-version ctx))})))

;; =========================== Stream Management ================================

(defn stream-flag [hstream]
  (let [res (int-array 1)]
    (with-check (JCudaDriver/cuStreamGetFlags (extract  hstream) res) (aget ^ints res 0))))

(defn stream-priority ^long [hstream]
  (let [res (int-array 1)]
    (with-check (JCudaDriver/cuStreamGetPriority (extract hstream) res) (aget ^ints res 0))))

(extend-type CUStream
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

(defn max-threads-per-block ^long [^CUfunction function]
  (info-attribute* JCudaDriver/cuFuncGetAttribute function
                   CUfunction_attribute/CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK))

(defn shared-size ^long [^CUfunction function]
  (info-attribute* JCudaDriver/cuFuncGetAttribute function
                   CUfunction_attribute/CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES))

(defn const-size ^long [^CUfunction function]
  (info-attribute* JCudaDriver/cuFuncGetAttribute function
                   CUfunction_attribute/CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES))

(defn local-size ^long [^CUfunction function]
  (info-attribute* JCudaDriver/cuFuncGetAttribute function
                   CUfunction_attribute/CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES))

(defn num-regs ^long [^CUfunction function]
  (info-attribute* JCudaDriver/cuFuncGetAttribute function
                   CUfunction_attribute/CU_FUNC_ATTRIBUTE_NUM_REGS))

(defn ptx-version ^long [^CUfunction function]
  (info-attribute* JCudaDriver/cuFuncGetAttribute function
                   CUfunction_attribute/CU_FUNC_ATTRIBUTE_PTX_VERSION))

(defn binary-version ^long [^CUfunction function]
  (info-attribute* JCudaDriver/cuFuncGetAttribute function
                   CUfunction_attribute/CU_FUNC_ATTRIBUTE_BINARY_VERSION))

(defn cache-config*
  "Sets the preferred cache configuration for a device function `fun`, as an integer `config`.

  See [cuFuncSetCacheConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
  "
  [fun ^long config]
  (with-check (JCudaDriver/cuFuncSetCacheConfig fun config) fun))

(defn cache-config!
  "Sets the preferred cache configuration for a device function `fun`, as a keyword `config`.

  Available configs are `:prefer-none`, `:prefer-shared`, `:prefer-L1`, and `:prefer-equal`.

  See [cuFuncSetCacheConfig](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
  "
  [fun config]
  (cache-config* fun (or (func-cache-config config)
                         (throw (ex-info "Invaling cache config."
                                         {:config config :available func-cache-config})))))

(extend-type CUfunction
  Info
  (info
    ([fun info-type]
     (maybe
      (case info-type
        :max-threads-per-block (max-threads-per-block fun)
        :shared-size (shared-size fun)
        :const-size (const-size fun)
        :local-size (local-size fun)
        :num-regs (num-regs fun)
        :ptx-version (ptx-version fun)
        :binary-version (binary-version fun)
        nil)))
    ([fun]
     {:max-threads-per-block (maybe (max-threads-per-block fun))
      :shared-size (maybe (shared-size fun))
      :const-size (maybe (const-size fun))
      :local-size (maybe (local-size fun))
      :num-regs (maybe (num-regs fun))
      :ptx-version (maybe (ptx-version fun))
      :binary-version (maybe (binary-version fun))})))
