---
title: "Get Started"
Author: Dragan Djuric
layout: article
---

ClojureCUDA uses native Nvidia GPU drivers, and CUDA toolkit, so it is very important that you do not skip any part of this guide.

## How to Get Started
* Walk through this guide, set up your development environment, and try the examples.
* Familiarize yourself with ClojureCUDA's [more detailed tutorials](/articles/guides.html) and [API documentation](/codox).

## Usage

First `use` or `require` `uncomplicate.clojurecuda.core` and/or `uncomplicate.clojurecuda.info` and/or `uncomplicate.clojurecuda.nvrtc`  in your namespace, and you'll be able to call appropriate functions from the ClojureCUDA library.

```clojure
(ns example
  (:use [uncomplicate.clojurecuda core info nvrtc]))
```

Now you can work with CUDA devices, contexts, streams, memory etc.

Here we initialize cuda and get the info of all devices.

```clojure
(init)
(map info (map device (range (device-count))))
```

## Overview and Features

ClojureCUDA is a Clojure library for High Performance Computing with CUDA, which supports Nvidia's GPUs. If you need to create programs for AMD, Intel, or even Nvidia's GPUs, or Intel's and AMD's CPUs, you probably need [ClojureCL](http://clojurecl.uncomplicate.org), ClojureCUDA's OpenCL based cousin.

If you need higher-level high performance functionality, such as matrix computations, try [Neanderthal](http://neanderthal.uncomplicate.org).

## Installation

### Install CUDA Toolkit

To use ClojureCUDA, you must have an Nvidia GPU, and install appropriate GPU drivers. If you need to create your own CUDA kernels (you most probably do), you also need CUDA Toolkit. You can download both the drivers and the toolkit as one bundle from [Nvidia's CUDA Toolkit page](https://developer.nvidia.com/cuda-toolkit). *Please note that ClojureCUDA requires a minimal CUDA version, which is currently 8.0, so make sure that you have recently updated your drivers and the toolkit.* If you use older drivers, some things might work, but some might not.

### Add ClojureCUDA jar

The most straightforward way to include ClojureCUDA in your project is with Leiningen. Add the following dependency to your `project.clj`:

![](http://clojars.org/uncomplicate/clojurecuda/latest-version.svg)

ClojureCUDA currently works out of the box on Linux, Windows, and OS X. For other plaforms, contact us.

## Where to go next

Hopefully this guide got you started and now you'd like to learn more. CUDA programming requires a lot of knowledge about the CUDA parallel computing model, devices and specifics of parallel computations. The best beginner's guide, in my opinion, is the [OpenCL in Action](http://www.amazon.com/OpenCL-Action-Accelerate-Graphics-Computations/dp/1617290173) book. It is not based on CUDA, but OpenCL (which is an open standard similar to CUDA supported by [ClojureCL](http://clojurecl.uncomplicate.org). Most books for CUDA are not as good as that one, but there are plenty of them, so you'll find the one that suits your tastes (but I don't know which one to recommend for beginners). I expect to build a comprehensive base of articles and references for exploring this topic, so please check the [All Guides](/articles/guides.html) page from time to time. Of course, you should also check the [ClojureCUDA API](/codox) for specific details, and feel free to take a glance at [the source](https://github.com/uncomplicate/clojurecuda) while you are there.
