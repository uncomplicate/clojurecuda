---
title: "Get Started"
Author: Dragan Djuric
layout: article
---

ClojureCUDA uses native Nvidia GPU drivers, and CUDA toolkit, so it is very important that you do not skip any part of this guide.

## How to Get Started
* Walk through this guide, set up your development environment, and try the examples.
* Familiarize yourself with ClojureCUDA's [more detailed tutorials](/articles/guides.html) and [API documentation](/codox).

## Minimum requirements
* Java 8 (but the newer the better).
* CUDA Drivers
* Linux or Windows. macOS doesn't allow CUDA from version 11 and up. You can only use an old release of ClojureCUDA on macOS.

## Installation

### Include the ClojureCUDA jar

The most straightforward way to include ClojureCUDA in your project is with Leiningen.

* Add the following dependency to your `project.clj`:![](https://clojars.org/uncomplicate/clojurecuda/latest-version.svg)
* Add the appropriate JavaCPP CUDA distribution jar, such as `[org.bytedeco/cuda "12.3-8.9-1.5.10" :classifier linux-x86_64-redist]`

If you use the latest CUDA (as of this writing, `12.3`) that's all. Please not that JavaCPP CUDA is *VERY LARGE*, so the download will take time
the first time you're doing it. If you do this from an IDE, you would not even know why your REPL is not up yet, and may kill the process. This
will leave the JavaCPP CUDA jar broken. So, the first time you're using this dependency, I advise you to open the terminal (command prompt on Windows)
and type `lein deps`. You'll see the progress and can patiently wait a minute or two until it's ready. The next time, your REPL will start instantly,
because everything will be cached in your local Maven repository (`<home>/.m2`). If you already messed it up, do not worry. Just go to `<home>/.m2/repository/org/bytedeco` and delete all folders that mention cuda.

ClojureCUDA currently works out of the box on Linux and Windows, while Nvidia does not support macOS. For other plaforms, contact us.

### Install CUDA Toolkit (LEGACY)

**This is only required for old ClojureCUDA versions (0.17.0 and older). For 0.18.0 and up, you only need to have recent Nvidia GPU drivers installed on your system.**

To use ClojureCUDA, you must have an Nvidia GPU, and install appropriate GPU drivers. If you need to create your own CUDA kernels (you most probably do), you also need CUDA Toolkit. You can download both the drivers and the toolkit as one bundle from [Nvidia's CUDA Toolkit page](https://developer.nvidia.com/cuda-toolkit). *Please note that ClojureCUDA requires a minimal CUDA version, which is currently `11.0`, and prefers the latest CUDA (currently `11.4`) so make sure that you have recently updated your drivers and the toolkit.* If you use older drivers, some things might work, but some might not.


## Usage

First `use` or `require` `uncomplicate.clojurecuda.core` and/or `uncomplicate.commons.core` and/or `uncomplicate.clojurecuda.info` in your namespace, and you'll be able to call appropriate functions from the ClojureCUDA library.

```clojure
(require '[uncomplicate.clojurecuda.core :refer :all]
         '[uncomplicate.commons.core :refer [info]]
         '[uncomplicate.clojurecuda.info :refer :all])
```

Now you can work with CUDA devices, contexts, streams, memory etc.

Here we initialize cuda and get the info of all devices.

```clojure
(init)
(map info (map device (range (device-count))))
```

If at least one device was found, you can continue with the following to verify that everything works on your system. This has been taken from [an introductory article](https://dragan.rocks/articles/18/Interactive-GPU-Programming-1-Hello-CUDA) and is explained in much more detail there if you want to know what's going on there:

```clojure
(def my-nvidia-gpu (device 0))
(def ctx (context my-nvidia-gpu))

;; set the current context
(current-context! ctx)

;; allocate memory on the GPU
(def gpu-array (mem-alloc 1024))

;; allocate memory on the host
(def main-array (float-array (range 256)))

;; copy host memory to the device
(memcpy-host! main-array gpu-array)

(def kernel-source
      "extern \"C\"
         __global__ void increment (int n, float *a) {
           int i = blockIdx.x * blockDim.x + threadIdx.x;
           if (i < n) {
             a[i] = a[i] + 1.0f;
        }
       };")

(def hello-program (compile! (program kernel-source)))
(def hello-module (module hello-program))
(def increment (function hello-module "increment"))
(launch! increment (grid-1d 256) (parameters 256 gpu-array))
(def result (memcpy-host! gpu-array (float-array 256)))

(take 12 result)
```

## Overview and Features

ClojureCUDA is a Clojure library for High Performance Computing with CUDA, which supports Nvidia's GPUs. If you need to create programs for AMD, Intel, or even Nvidia's GPUs, or Intel's and AMD's CPUs, you probably need [ClojureCL](https://clojurecl.uncomplicate.org), ClojureCUDA's OpenCL based cousin.

If you need higher-level high performance functionality, such as matrix computations, try [Neanderthal](https://neanderthal.uncomplicate.org).


## Where to go next

Hopefully this guide got you started and now you'd like to learn more. CUDA programming requires a lot of knowledge about the CUDA parallel computing model, devices and specifics of parallel computations. The best beginner's guide, in my opinion, is the [OpenCL in Action](https://www.amazon.com/OpenCL-Action-Accelerate-Graphics-Computations/dp/1617290173) book. It is not based on CUDA, but OpenCL (which is an open standard similar to CUDA supported by [ClojureCL](https://clojurecl.uncomplicate.org). Most books for CUDA are not as good as that one, but there are plenty of them, so you'll find the one that suits your tastes (but I don't know which one to recommend for beginners). I expect to build a comprehensive base of articles and references for exploring this topic, so please check the [All Guides](/articles/guides.html) page from time to time. Of course, you should also check the [ClojureCUDA API](/codox) for specific details, and feel free to take a glance at [the source](https://github.com/uncomplicate/clojurecuda) while you are there.
