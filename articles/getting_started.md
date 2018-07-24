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
* Java 8
* libstdc++ 3.4.21 (comes with GCC 5)
* CUDA Toolkit 9

## Usage

First `use` or `require` `uncomplicate.clojurecuda.core` and/or `uncomplicate.clojurecuda.info` and/or `uncomplicate.clojurecuda.nvrtc`  in your namespace, and you'll be able to call appropriate functions from the ClojureCUDA library.

```clojure
(require '[uncomplicate.clojurecuda.core :refer :all]
         '[uncomplicate.clojurecuda.info :refer :all]
         '[uncomplicate.clojurecuda.nvrtc :refer :all])
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

## Installation

### Install CUDA Toolkit

To use ClojureCUDA, you must have an Nvidia GPU, and install appropriate GPU drivers. If you need to create your own CUDA kernels (you most probably do), you also need CUDA Toolkit. You can download both the drivers and the toolkit as one bundle from [Nvidia's CUDA Toolkit page](https://developer.nvidia.com/cuda-toolkit). *Please note that ClojureCUDA requires a minimal CUDA version, which is currently `9.0`, and prefers the latest CUDA (currently `9.2`) so make sure that you have recently updated your drivers and the toolkit.* If you use older drivers, some things might work, but some might not.

### Add ClojureCUDA jar

The most straightforward way to include ClojureCUDA in your project is with Leiningen. Add the following dependency to your `project.clj`:

![](https://clojars.org/uncomplicate/clojurecuda/latest-version.svg)

If you use the latest CUDA (as of this writing, `9.2`) that's all. If you must use CUDA `9.0`, or `9.1`, add an explicit
dependency to `jcuda/jcuda` `0.9.0b`, or `0.9.1` to your project(s).

ClojureCUDA currently works out of the box on Linux, Windows, and OS X. For other plaforms, contact us.

## Where to go next

Hopefully this guide got you started and now you'd like to learn more. CUDA programming requires a lot of knowledge about the CUDA parallel computing model, devices and specifics of parallel computations. The best beginner's guide, in my opinion, is the [OpenCL in Action](https://www.amazon.com/OpenCL-Action-Accelerate-Graphics-Computations/dp/1617290173) book. It is not based on CUDA, but OpenCL (which is an open standard similar to CUDA supported by [ClojureCL](https://clojurecl.uncomplicate.org). Most books for CUDA are not as good as that one, but there are plenty of them, so you'll find the one that suits your tastes (but I don't know which one to recommend for beginners). I expect to build a comprehensive base of articles and references for exploring this topic, so please check the [All Guides](/articles/guides.html) page from time to time. Of course, you should also check the [ClojureCUDA API](/codox) for specific details, and feel free to take a glance at [the source](https://github.com/uncomplicate/clojurecuda) while you are there.
