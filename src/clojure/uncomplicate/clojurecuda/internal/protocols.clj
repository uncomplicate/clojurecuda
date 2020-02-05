;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.clojurecuda.internal.protocols)

;; =================== Object wrappers =================================

(defprotocol Mem
  "An object that represents memory that participates in CUDA operations.
  It can be on the device, or on the host.  Built-in implementations:
  cuda pointers, Java primitive arrays and ByteBuffers"
  (ptr [this] "`Pointer` to this object.")
  (size [this] "Memory size of this cuda or host object in bytes.")
  (memcpy-host* [this host size] [this host size hstream]))

(defprotocol HostMem
  (host-ptr [this] "Host `Pointer` to this object.")
  (host-buffer [this] "The actual `ByteBuffer` on the host"))

(defprotocol JITOption
  (put-jit-option [value option options]))

(defprotocol ModuleLoad
  (module-load* [data m])
  (link-add* [data link-state type options]))

(defprotocol WithOffset
  (with-offset [this ofst]))
