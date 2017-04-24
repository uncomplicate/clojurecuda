---
title: "Guides"
Author: Dragan Djuric
layout: article
---

## Making sense of CUDA

[CUDA](http://www.nvidia.com/object/cuda_home_new.html) is similar to [OpenCL](https://www.khronos.org/opencl/), a standard for heterogeneous parallel computing and GPGPU (see more about that at [ClojureCL homepage](http://clojurecl.uncomplicate.org). While OpenCL is an open standard that is supported on
a multitude of hardware platforms, CUDA runs only on Nvidia GPUs. Why is CUDA useful, then? Because, unlike OpenCL, CUDA gets lots of resource investment from Nvidia, which spends lots of money developing a bunch of useful libraries, most notably cuBLAS (matrices and linear algebra), cuFFT (fast furier transforms) and cuDNN (neural networks and deep learning). Additionally, major cloud providers offer Nvidia GPUs in the cloud, while AMD GPUs are much harder to find there. The C++ tooling for CUDA is also much more featureful than OpenCL, but that is not an issue here, since in Clojure both are equally easy to develop (although I like the OpenCL way more).

CUDA brings a lot of power, but do not expect it to be an easy ride if you've never programmed anything on the GPU or embedded devices. With ClojureCUDA, it is not as difficult as in C++, but you still have to grasp the concepts of parallel programming that are different than your usual x86 CPU Java, C, Clojure, C#, Python or Ruby code. **The good news is that you can use any CUDA book to learn ClojureCUDA.

Once you get past the beginner's steep learning curve, it makes sense, and opens a whole new world of high-performance
computing - you practically have a supercomputer on your desktop.

Especially at the beginning, you might like to also try [ClojureCL](http://clojurecl.uncomplicate.org), since it is a more mature project with more examples, and even most of the book [OpenCL in Action](http://www.amazon.com/OpenCL-Action-Accelerate-Graphics-Computations/dp/1617290173) worked out in ClojureCL. That knowledge (and examples) are relatively easily transferrable to ClojureCUDA.

## Where to find CUDA books, tutorials, and documentation

Learning CUDA programming requires learning the details of CUDA C++ kernel language and CUDA driver API, but even more important is learning the main concept of high performance computing, generally applicable in OpenCL, CUDA, Open MPI
or other technologies.

1. In my opinion, [OpenCL in Action](http://www.amazon.com/OpenCL-Action-Accelerate-Graphics-Computations/dp/1617290173) is by far the best, especially if you're a beginner. You could try it first despite it not being CUDA-based, or you might try some of CUDA books **aimed at beginners**. If you've never programmed GPUs or other massively parallel processors, do not skip this first step!
2. When you get past the beginning, you'll probably need a specific optimization guide for the specific Nvidia GPU architecture you use. Nvidia offers up-to-date guides, and I've found that [CUDA Handbook: A Comprehensive Guide to GPU Programming](https://www.amazon.com/CUDA-Handbook-Comprehensive-Guide-Programming/dp/0321809467) is a great advanced resource.
3. [CUDA Driver API Specification](http://docs.nvidia.com/cuda/cuda-driver-api/index.html) is what you need when you need to look up some specific of a certain ClojureCUDA function. Of course, after you've tried [ClojureCUDA API documentation](/codox) ;)
4. Algorithms for parallel computations are generally different than classic algorithms from the textbook, and are
usually platform-agnostic. You'll usually find a solution to your computation problem in a scientific paper or a general HPC book regardless of whether it is written for OpenCL, CUDA, or is thechnology neutral.

## ClojureCUDA Reference

1. ClojureCUDA comes with [detailed documentation](/codox). Be sure to check it, it also includes examples and foreign links.
2. ClojureCUDA comes with a bunch of [Midje tests](https://github.com/uncomplicate/clojurecuda/tree/master/test/clojure/uncomplicate/clojurecuda/). When you're not sure how to use some feature, consult the tests.
