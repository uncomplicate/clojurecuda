/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__DRIVER_TYPES_H__)
#define __DRIVER_TYPES_H__

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__
#endif

#ifndef __DOXYGEN_ONLY__
#include "crt/host_defines.h"
#endif
#include "vector_types.h"



#ifndef __CUDACC_RTC_MINIMAL__
/**
 * \defgroup CUDART_TYPES Data types used by CUDA Runtime
 * \ingroup CUDART
 *
 * @{
 */

/*******************************************************************************
*                                                                              *
*  TYPE DEFINITIONS USED BY RUNTIME API                                        *
*                                                                              *
*******************************************************************************/

#if !defined(__CUDA_INTERNAL_COMPILATION__)


#if !defined(__CUDACC_RTC__)
#include <limits.h>
#include <stddef.h>
#endif /* !defined(__CUDACC_RTC__) */

#define cudaHostAllocDefault                0x00  /**< Default page-locked allocation flag */
#define cudaHostAllocPortable               0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostAllocMapped                 0x02  /**< Map allocation into device space */
#define cudaHostAllocWriteCombined          0x04  /**< Write-combined memory */

#define cudaHostRegisterDefault             0x00  /**< Default host memory registration flag */
#define cudaHostRegisterPortable            0x01  /**< Pinned memory accessible by all CUDA contexts */
#define cudaHostRegisterMapped              0x02  /**< Map registered memory into device space */
#define cudaHostRegisterIoMemory            0x04  /**< Memory-mapped I/O space */
#define cudaHostRegisterReadOnly            0x08  /**< Memory-mapped read-only */

#define cudaPeerAccessDefault               0x00  /**< Default peer addressing enable flag */

#define cudaStreamDefault                   0x00  /**< Default stream flag */
#define cudaStreamNonBlocking               0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */

 /**
 * Legacy stream handle
 *
 * Stream handle that can be passed as a cudaStream_t to use an implicit stream
 * with legacy synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define cudaStreamLegacy                    ((cudaStream_t)0x1)

/**
 * Per-thread stream handle
 *
 * Stream handle that can be passed as a cudaStream_t to use an implicit stream
 * with per-thread synchronization behavior.
 *
 * See details of the \link_sync_behavior
 */
#define cudaStreamPerThread                 ((cudaStream_t)0x2)

#define cudaEventDefault                    0x00  /**< Default event flag */
#define cudaEventBlockingSync               0x01  /**< Event uses blocking synchronization */
#define cudaEventDisableTiming              0x02  /**< Event will not record timing data */
#define cudaEventInterprocess               0x04  /**< Event is suitable for interprocess use. cudaEventDisableTiming must be set */

#define cudaEventRecordDefault              0x00  /**< Default event record flag */
#define cudaEventRecordExternal             0x01  /**< Event is captured in the graph as an external event node when performing stream capture */

#define cudaEventWaitDefault                0x00  /**< Default event wait flag */
#define cudaEventWaitExternal               0x01  /**< Event is captured in the graph as an external event node when performing stream capture */

#define cudaDeviceScheduleAuto              0x00  /**< Device flag - Automatic scheduling */
#define cudaDeviceScheduleSpin              0x01  /**< Device flag - Spin default scheduling */
#define cudaDeviceScheduleYield             0x02  /**< Device flag - Yield default scheduling */
#define cudaDeviceScheduleBlockingSync      0x04  /**< Device flag - Use blocking synchronization */
#define cudaDeviceBlockingSync              0x04  /**< Device flag - Use blocking synchronization 
                                                    *  \deprecated This flag was deprecated as of CUDA 4.0 and
                                                    *  replaced with ::cudaDeviceScheduleBlockingSync. */
#define cudaDeviceScheduleMask              0x07  /**< Device schedule flags mask */
#define cudaDeviceMapHost                   0x08  /**< Device flag - Support mapped pinned allocations */
#define cudaDeviceLmemResizeToMax           0x10  /**< Device flag - Keep local memory allocation after launch */
#define cudaDeviceSyncMemops                0x80  /**< Device flag - Ensure synchronous memory operations on this context will synchronize */
#define cudaDeviceMask                      0xff  /**< Device flags mask */

#define cudaArrayDefault                    0x00  /**< Default CUDA array allocation flag */
#define cudaArrayLayered                    0x01  /**< Must be set in cudaMalloc3DArray to create a layered CUDA array */
#define cudaArraySurfaceLoadStore           0x02  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to bind surfaces to the CUDA array */
#define cudaArrayCubemap                    0x04  /**< Must be set in cudaMalloc3DArray to create a cubemap CUDA array */
#define cudaArrayTextureGather              0x08  /**< Must be set in cudaMallocArray or cudaMalloc3DArray in order to perform texture gather operations on the CUDA array */
#define cudaArrayColorAttachment            0x20  /**< Must be set in cudaExternalMemoryGetMappedMipmappedArray if the mipmapped array is used as a color target in a graphics API */
#define cudaArraySparse                     0x40  /**< Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a sparse CUDA array or CUDA mipmapped array */
#define cudaArrayDeferredMapping            0x80  /**< Must be set in cudaMallocArray, cudaMalloc3DArray or cudaMallocMipmappedArray in order to create a deferred mapping CUDA array or CUDA mipmapped array */

#define cudaIpcMemLazyEnablePeerAccess      0x01  /**< Automatically enable peer access between remote devices as needed */

#define cudaMemAttachGlobal                 0x01  /**< Memory can be accessed by any stream on any device*/
#define cudaMemAttachHost                   0x02  /**< Memory cannot be accessed by any stream on any device */
#define cudaMemAttachSingle                 0x04  /**< Memory can only be accessed by a single stream on the associated device */

#define cudaOccupancyDefault                0x00  /**< Default behavior */
#define cudaOccupancyDisableCachingOverride 0x01  /**< Assume global caching is enabled and cannot be automatically turned off */

#define cudaCpuDeviceId                     ((int)-1) /**< Device id that represents the CPU */
#define cudaInvalidDeviceId                 ((int)-2) /**< Device id that represents an invalid device */
#define cudaInitDeviceFlagsAreValid         0x01  /**< Tell the CUDA runtime that DeviceFlags is being set in cudaInitDevice call */

#endif /* !__CUDA_INTERNAL_COMPILATION__ */

/** \cond impl_private */
#if defined(__DOXYGEN_ONLY__) || defined(CUDA_ENABLE_DEPRECATED)
#define __CUDA_DEPRECATED
#elif defined(_MSC_VER)
#define __CUDA_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __CUDA_DEPRECATED __attribute__((deprecated))
#else
#define __CUDA_DEPRECATED
#endif
/** \endcond impl_private */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

/**
 * CUDA error types
 */
enum __device_builtin__ cudaError
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * also means that the operation being queried is complete (see
     * ::cudaEventQuery() and ::cudaStreamQuery()).
     */
    cudaSuccess                           =      0,
  
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    cudaErrorInvalidValue                 =     1,
  
    /**
     * The API call failed because it was unable to allocate enough memory or
     * other resources to perform the requested operation.
     */
    cudaErrorMemoryAllocation             =      2,
  
    /**
     * The API call failed because the CUDA driver and runtime could not be
     * initialized.
     */
    cudaErrorInitializationError          =      3,
  
    /**
     * This indicates that a CUDA Runtime API call cannot be executed because
     * it is being called during process shut down, at a point in time after
     * CUDA driver has been unloaded.
     */
    cudaErrorCudartUnloading              =     4,

    /**
     * This indicates profiler is not initialized for this run. This can
     * happen when the application is running with external profiling tools
     * like visual profiler.
     */
    cudaErrorProfilerDisabled             =     5,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to attempt to enable/disable the profiling via ::cudaProfilerStart or
     * ::cudaProfilerStop without initialization.
     */
    cudaErrorProfilerNotInitialized       =     6,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStart() when profiling is already enabled.
     */
    cudaErrorProfilerAlreadyStarted       =     7,

    /**
     * \deprecated
     * This error return is deprecated as of CUDA 5.0. It is no longer an error
     * to call cudaProfilerStop() when profiling is already disabled.
     */
     cudaErrorProfilerAlreadyStopped       =    8,
    /**
     * This indicates that a kernel launch is requesting resources that can
     * never be satisfied by the current device. Requesting more shared memory
     * per block than the device supports will trigger this error, as will
     * requesting too many threads or blocks. See ::cudaDeviceProp for more
     * device limitations.
     */
    cudaErrorInvalidConfiguration         =      9,
  
    /**
     * This indicates that one or more of the pitch-related parameters passed
     * to the API call is not within the acceptable range for pitch.
     */
    cudaErrorInvalidPitchValue            =     12,
  
    /**
     * This indicates that the symbol name/identifier passed to the API call
     * is not a valid name or identifier.
     */
    cudaErrorInvalidSymbol                =     13,

    /**
     * This indicates that at least one host pointer passed to the API call is
     * not a valid host pointer.
     * \deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    cudaErrorInvalidHostPointer           =     16,
  
    /**
     * This indicates that at least one device pointer passed to the API call is
     * not a valid device pointer.
     * \deprecated
     * This error return is deprecated as of CUDA 10.1.
     */
    cudaErrorInvalidDevicePointer         =     17,
    /**
     * This indicates that the texture passed to the API call is not a valid
     * texture.
     */
    cudaErrorInvalidTexture               =     18,
  
    /**
     * This indicates that the texture binding is not valid. This occurs if you
     * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
     */
    cudaErrorInvalidTextureBinding        =     19,
  
    /**
     * This indicates that the channel descriptor passed to the API call is not
     * valid. This occurs if the format is not one of the formats specified by
     * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
     */
    cudaErrorInvalidChannelDescriptor     =     20,
  
    /**
     * This indicates that the direction of the memcpy passed to the API call is
     * not one of the types specified by ::cudaMemcpyKind.
     */
    cudaErrorInvalidMemcpyDirection       =     21,

    /**
     * This indicated that the user has taken the address of a constant variable,
     * which was forbidden up until the CUDA 3.1 release.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Variables in constant
     * memory may now have their address taken by the runtime via
     * ::cudaGetSymbolAddress().
     */
    cudaErrorAddressOfConstant            =     22,
  
    /**
     * This indicated that a texture fetch was not able to be performed.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureFetchFailed           =     23,
  
    /**
     * This indicated that a texture was not bound for access.
     * This was previously used for device emulation of texture operations.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorTextureNotBound              =     24,
  
    /**
     * This indicated that a synchronization operation had failed.
     * This was previously used for some device emulation functions.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorSynchronizationError         =     25,
    /**
     * This indicates that a non-float texture was being accessed with linear
     * filtering. This is not supported by CUDA.
     */
    cudaErrorInvalidFilterSetting         =     26,
  
    /**
     * This indicates that an attempt was made to read an unsupported data type as a
     * normalized float. This is not supported by CUDA.
     */
    cudaErrorInvalidNormSetting           =     27,

    /**
     * Mixing of device and device emulation code was not allowed.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMixedDeviceExecution         =     28,

    /**
     * This indicates that the API call is not yet implemented. Production
     * releases of CUDA will never return this error.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    cudaErrorNotYetImplemented            =     31,
  
    /**
     * This indicated that an emulated device pointer exceeded the 32-bit address
     * range.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorMemoryValueTooLarge          =     32,
    /**
     * This indicates that the CUDA driver that the application has loaded is a
     * stub library. Applications that run with the stub rather than a real
     * driver loaded will result in CUDA API returning this error.
     */
    cudaErrorStubLibrary                  =     34,

    /**
     * This indicates that the installed NVIDIA CUDA driver is older than the
     * CUDA runtime library. This is not a supported configuration. Users should
     * install an updated NVIDIA display driver to allow the application to run.
     */
    cudaErrorInsufficientDriver           =     35,

    /**
     * This indicates that the API call requires a newer CUDA driver than the one
     * currently installed. Users should install an updated NVIDIA CUDA driver
     * to allow the API call to succeed.
     */
    cudaErrorCallRequiresNewerDriver      =     36,
  
    /**
     * This indicates that the surface passed to the API call is not a valid
     * surface.
     */
    cudaErrorInvalidSurface               =     37,
  
    /**
     * This indicates that multiple global or constant variables (across separate
     * CUDA source files in the application) share the same string name.
     */
    cudaErrorDuplicateVariableName        =     43,
  
    /**
     * This indicates that multiple textures (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateTextureName         =     44,
  
    /**
     * This indicates that multiple surfaces (across separate CUDA source
     * files in the application) share the same string name.
     */
    cudaErrorDuplicateSurfaceName         =     45,
  
    /**
     * This indicates that all CUDA devices are busy or unavailable at the current
     * time. Devices are often busy/unavailable due to use of
     * ::cudaComputeModeProhibited, ::cudaComputeModeExclusiveProcess, or when long
     * running CUDA kernels have filled up the GPU and are blocking new work
     * from starting. They can also be unavailable due to memory constraints
     * on a device that already has active CUDA work being performed.
     */
    cudaErrorDevicesUnavailable           =     46,
  
    /**
     * This indicates that the current context is not compatible with this
     * the CUDA Runtime. This can only occur if you are using CUDA
     * Runtime/Driver interoperability and have created an existing Driver
     * context using the driver API. The Driver context may be incompatible
     * either because the Driver context was created using an older version 
     * of the API, because the Runtime API call expects a primary driver 
     * context and the Driver context is not primary, or because the Driver 
     * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions 
     * with the CUDA Driver API" for more information.
     */
    cudaErrorIncompatibleDriverContext    =     49,
    
    /**
     * The device function being invoked (usually via ::cudaLaunchKernel()) was not
     * previously configured via the ::cudaConfigureCall() function.
     */
    cudaErrorMissingConfiguration         =      52,

    /**
     * This indicated that a previous kernel launch failed. This was previously
     * used for device emulation of kernel launches.
     * \deprecated
     * This error return is deprecated as of CUDA 3.1. Device emulation mode was
     * removed with the CUDA 3.1 release.
     */
    cudaErrorPriorLaunchFailure           =      53,
    /**
     * This error indicates that a device runtime grid launch did not occur 
     * because the depth of the child grid would exceed the maximum supported
     * number of nested grid launches. 
     */
    cudaErrorLaunchMaxDepthExceeded       =     65,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped textures which are unsupported by the device runtime. 
     * Kernels launched via the device runtime only support textures created with 
     * the Texture Object API's.
     */
    cudaErrorLaunchFileScopedTex          =     66,

    /**
     * This error indicates that a grid launch did not occur because the kernel 
     * uses file-scoped surfaces which are unsupported by the device runtime.
     * Kernels launched via the device runtime only support surfaces created with
     * the Surface Object API's.
     */
    cudaErrorLaunchFileScopedSurf         =     67,

    /**
     * This error indicates that a call to ::cudaDeviceSynchronize made from
     * the device runtime failed because the call was made at grid depth greater
     * than than either the default (2 levels of grids) or user specified device
     * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on
     * launched grids at a greater depth successfully, the maximum nested
     * depth at which ::cudaDeviceSynchronize will be called must be specified
     * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
     * api before the host-side launch of a kernel using the device runtime.
     * Keep in mind that additional levels of sync depth require the runtime
     * to reserve large amounts of device memory that cannot be used for
     * user allocations. Note that ::cudaDeviceSynchronize made from device
     * runtime is only supported on devices of compute capability < 9.0.
     */
    cudaErrorSyncDepthExceeded            =     68,

    /**
     * This error indicates that a device runtime grid launch failed because
     * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
     * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
     * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher 
     * than the upper bound of outstanding launches that can be issued to the
     * device runtime. Keep in mind that raising the limit of pending device
     * runtime launches will require the runtime to reserve device memory that
     * cannot be used for user allocations.
     */
    cudaErrorLaunchPendingCountExceeded   =     69,
  
    /**
     * The requested device function does not exist or is not compiled for the
     * proper device architecture.
     */
    cudaErrorInvalidDeviceFunction        =      98,
  
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    cudaErrorNoDevice                     =     100,
  
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device or that the action requested is
     * invalid for the specified device.
     */
    cudaErrorInvalidDevice                =     101,

    /**
     * This indicates that the device doesn't have a valid Grid License.
     */
    cudaErrorDeviceNotLicensed            =     102,

   /**
    * By default, the CUDA runtime may perform a minimal set of self-tests,
    * as well as CUDA driver tests, to establish the validity of both.
    * Introduced in CUDA 11.2, this error return indicates that at least one
    * of these tests has failed and the validity of either the runtime
    * or the driver could not be established.
    */
   cudaErrorSoftwareValidityNotEstablished  =     103,

    /**
     * This indicates an internal startup failure in the CUDA runtime.
     */
    cudaErrorStartupFailure               =    127,
  
    /**
     * This indicates that the device kernel image is invalid.
     */
    cudaErrorInvalidKernelImage           =     200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    cudaErrorDeviceUninitialized          =     201,

    /**
     * This indicates that the buffer object could not be mapped.
     */
    cudaErrorMapBufferObjectFailed        =     205,
  
    /**
     * This indicates that the buffer object could not be unmapped.
     */
    cudaErrorUnmapBufferObjectFailed      =     206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    cudaErrorArrayIsMapped                =     207,

    /**
     * This indicates that the resource is already mapped.
     */
    cudaErrorAlreadyMapped                =     208,
  
    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    cudaErrorNoKernelImageForDevice       =     209,

    /**
     * This indicates that a resource has already been acquired.
     */
    cudaErrorAlreadyAcquired              =     210,

    /**
     * This indicates that a resource is not mapped.
     */
    cudaErrorNotMapped                    =     211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    cudaErrorNotMappedAsArray             =     212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    cudaErrorNotMappedAsPointer           =     213,
  
    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    cudaErrorECCUncorrectable             =     214,
  
    /**
     * This indicates that the ::cudaLimit passed to the API call is not
     * supported by the active device.
     */
    cudaErrorUnsupportedLimit             =     215,
    
    /**
     * This indicates that a call tried to access an exclusive-thread device that 
     * is already in use by a different thread.
     */
    cudaErrorDeviceAlreadyInUse           =     216,

    /**
     * This error indicates that P2P access is not supported across the given
     * devices.
     */
    cudaErrorPeerAccessUnsupported        =     217,

    /**
     * A PTX compilation failed. The runtime may fall back to compiling PTX if
     * an application does not contain a suitable binary for the current device.
     */
    cudaErrorInvalidPtx                   =     218,

    /**
     * This indicates an error with the OpenGL or DirectX context.
     */
    cudaErrorInvalidGraphicsContext       =     219,

    /**
     * This indicates that an uncorrectable NVLink error was detected during the
     * execution.
     */
    cudaErrorNvlinkUncorrectable          =     220,

    /**
     * This indicates that the PTX JIT compiler library was not found. The JIT Compiler
     * library is used for PTX compilation. The runtime may fall back to compiling PTX
     * if an application does not contain a suitable binary for the current device.
     */
    cudaErrorJitCompilerNotFound          =     221,

    /**
     * This indicates that the provided PTX was compiled with an unsupported toolchain.
     * The most common reason for this, is the PTX was generated by a compiler newer
     * than what is supported by the CUDA driver and PTX JIT compiler.
     */
    cudaErrorUnsupportedPtxVersion        =     222,

    /**
     * This indicates that the JIT compilation was disabled. The JIT compilation compiles
     * PTX. The runtime may fall back to compiling PTX if an application does not contain
     * a suitable binary for the current device.
     */
    cudaErrorJitCompilationDisabled       =     223,

    /**
     * This indicates that the provided execution affinity is not supported by the device.
     */
    cudaErrorUnsupportedExecAffinity      =     224,

    /**
     * This indicates that the code to be compiled by the PTX JIT contains
     * unsupported call to cudaDeviceSynchronize.
     */
    cudaErrorUnsupportedDevSideSync       =     225,

    /**
     * This indicates that an exception occurred on the device that is now
     * contained by the GPU's error containment capability. Common causes are -
     * a. Certain types of invalid accesses of peer GPU memory over nvlink
     * b. Certain classes of hardware errors
     * This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must
     * be terminated and relaunched.
     */
    cudaErrorContained                    =     226,

    /**
     * This indicates that the device kernel source is invalid.
     */
    cudaErrorInvalidSource                =     300,

    /**
     * This indicates that the file specified was not found.
     */
    cudaErrorFileNotFound                 =     301,
  
    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    cudaErrorSharedObjectSymbolNotFound   =     302,
  
    /**
     * This indicates that initialization of a shared object failed.
     */
    cudaErrorSharedObjectInitFailed       =     303,

    /**
     * This error indicates that an OS call failed.
     */
    cudaErrorOperatingSystem              =     304,
  
    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::cudaStream_t and
     * ::cudaEvent_t.
     */
    cudaErrorInvalidResourceHandle        =     400,

    /**
     * This indicates that a resource required by the API call is not in a
     * valid state to perform the requested operation.
     */
    cudaErrorIllegalState                 =     401,

    /**
     * This indicates an attempt was made to introspect an object in a way that
     * would discard semantically important information. This is either due to
     * the object using funtionality newer than the API version used to
     * introspect it or omission of optional return arguments.
     */
    cudaErrorLossyQuery                   =     402,

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, driver function names, texture names,
     * and surface names.
     */
    cudaErrorSymbolNotFound               =     500,
  
    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::cudaSuccess (which indicates completion). Calls that
     * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
     */
    cudaErrorNotReady                     =     600,

    /**
     * The device encountered a load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalAddress               =     700,
  
    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. Although this error is similar to
     * ::cudaErrorInvalidConfiguration, this error usually indicates that the
     * user has attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register count.
     */
    cudaErrorLaunchOutOfResources         =      701,
  
    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute 
     * \ref ::cudaDeviceAttr::cudaDevAttrKernelExecTimeout "cudaDevAttrKernelExecTimeout"
     * for more information.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchTimeout                =      702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    cudaErrorLaunchIncompatibleTexturing  =     703,
      
    /**
     * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
     * trying to re-enable peer addressing on from a context which has already
     * had peer addressing enabled.
     */
    cudaErrorPeerAccessAlreadyEnabled     =     704,
    
    /**
     * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to 
     * disable peer addressing which has not been enabled yet via 
     * ::cudaDeviceEnablePeerAccess().
     */
    cudaErrorPeerAccessNotEnabled         =     705,
  
    /**
     * This indicates that the user has called ::cudaSetValidDevices(),
     * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
     * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
     * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
     * calling non-device management operations (allocating memory and
     * launching kernels are examples of non-device management operations).
     * This error can also be returned if using runtime/driver
     * interoperability and there is an existing ::CUcontext active on the
     * host thread.
     */
    cudaErrorSetOnActiveProcess           =     708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    cudaErrorContextIsDestroyed           =     709,

    /**
     * An assert triggered in device code during kernel execution. The device
     * cannot be used again. All existing allocations are invalid. To continue
     * using CUDA, the process must be terminated and relaunched.
     */
    cudaErrorAssert                        =    710,
  
    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices 
     * passed to ::cudaEnablePeerAccess().
     */
    cudaErrorTooManyPeers                 =     711,
  
    /**
     * This error indicates that the memory range passed to ::cudaHostRegister()
     * has already been registered.
     */
    cudaErrorHostMemoryAlreadyRegistered  =     712,
        
    /**
     * This error indicates that the pointer passed to ::cudaHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    cudaErrorHostMemoryNotRegistered      =     713,

    /**
     * Device encountered an error in the call stack during kernel execution,
     * possibly due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorHardwareStackError           =     714,

    /**
     * The device encountered an illegal instruction during kernel execution
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorIllegalInstruction           =     715,

    /**
     * The device encountered a load or store instruction
     * on a memory address which is not aligned.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorMisalignedAddress            =     716,

    /**
     * While executing a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidAddressSpace          =     717,

    /**
     * The device encountered an invalid program counter.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorInvalidPc                    =     718,
  
    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. Less common cases can be system specific - more
     * information about these cases can be found in the system specific user guide.
     * This leaves the process in an inconsistent state and any further CUDA work
     * will return the same error. To continue using CUDA, the process must be terminated
     * and relaunched.
     */
    cudaErrorLaunchFailure                =      719,

    /**
     * This error indicates that the number of blocks launched per grid for a kernel that was
     * launched via either ::cudaLaunchCooperativeKernel
     * exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor
     * or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors
     * as specified by the device attribute ::cudaDevAttrMultiProcessorCount.
     */
    cudaErrorCooperativeLaunchTooLarge    =     720,

    /**
     * An exception occurred on the device while exiting a kernel using tensor memory: the
     * tensor memory was not completely deallocated. This leaves the process in an inconsistent
     * state and any further CUDA work will return the same error. To continue using CUDA, the
     * process must be terminated and relaunched.
     */
    cudaErrorTensorMemoryLeak             =     721,
    
    /**
     * This error indicates the attempted operation is not permitted.
     */
    cudaErrorNotPermitted                 =     800,

    /**
     * This error indicates the attempted operation is not supported
     * on the current system or device.
     */
    cudaErrorNotSupported                 =     801,

    /**
     * This error indicates that the system is not yet ready to start any CUDA
     * work.  To continue using CUDA, verify the system configuration is in a
     * valid state and all required driver daemons are actively running.
     * More information about this error can be found in the system specific
     * user guide.
     */
    cudaErrorSystemNotReady               =     802,

    /**
     * This error indicates that there is a mismatch between the versions of
     * the display driver and the CUDA driver. Refer to the compatibility documentation
     * for supported versions.
     */
    cudaErrorSystemDriverMismatch         =     803,

    /**
     * This error indicates that the system was upgraded to run with forward compatibility
     * but the visible hardware detected by CUDA does not support this configuration.
     * Refer to the compatibility documentation for the supported hardware matrix or ensure
     * that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES
     * environment variable.
     */
    cudaErrorCompatNotSupportedOnDevice   =     804,

    /**
     * This error indicates that the MPS client failed to connect to the MPS control daemon or the MPS server.
     */
    cudaErrorMpsConnectionFailed          =     805,

    /**
     * This error indicates that the remote procedural call between the MPS server and the MPS client failed.
     */
    cudaErrorMpsRpcFailure                =     806,

    /**
     * This error indicates that the MPS server is not ready to accept new MPS client requests.
     * This error can be returned when the MPS server is in the process of recovering from a fatal failure.
     */
    cudaErrorMpsServerNotReady            =     807,

    /**
     * This error indicates that the hardware resources required to create MPS client have been exhausted.
     */
    cudaErrorMpsMaxClientsReached         =     808,

    /**
     * This error indicates the the hardware resources required to device connections have been exhausted.
     */
    cudaErrorMpsMaxConnectionsReached     =     809,

    /**
     * This error indicates that the MPS client has been terminated by the server. To continue using CUDA, the process must be terminated and relaunched.
     */
    cudaErrorMpsClientTerminated          =     810,

    /**
     * This error indicates, that the program is using CUDA Dynamic Parallelism, but the current configuration, like MPS, does not support it.
     */
    cudaErrorCdpNotSupported              =     811,

    /**
     * This error indicates, that the program contains an unsupported interaction between different versions of CUDA Dynamic Parallelism.
     */
    cudaErrorCdpVersionMismatch           =     812,

    /**
     * The operation is not permitted when the stream is capturing.
     */
    cudaErrorStreamCaptureUnsupported     =    900,

    /**
     * The current capture sequence on the stream has been invalidated due to
     * a previous error.
     */
    cudaErrorStreamCaptureInvalidated     =    901,

    /**
     * The operation would have resulted in a merge of two independent capture
     * sequences.
     */
    cudaErrorStreamCaptureMerge           =    902,

    /**
     * The capture was not initiated in this stream.
     */
    cudaErrorStreamCaptureUnmatched       =    903,

    /**
     * The capture sequence contains a fork that was not joined to the primary
     * stream.
     */
    cudaErrorStreamCaptureUnjoined        =    904,

    /**
     * A dependency would have been created which crosses the capture sequence
     * boundary. Only implicit in-stream ordering dependencies are allowed to
     * cross the boundary.
     */
    cudaErrorStreamCaptureIsolation       =    905,

    /**
     * The operation would have resulted in a disallowed implicit dependency on
     * a current capture sequence from cudaStreamLegacy.
     */
    cudaErrorStreamCaptureImplicit        =    906,

    /**
     * The operation is not permitted on an event which was last recorded in a
     * capturing stream.
     */
    cudaErrorCapturedEvent                =    907,
  
    /**
     * A stream capture sequence not initiated with the ::cudaStreamCaptureModeRelaxed
     * argument to ::cudaStreamBeginCapture was passed to ::cudaStreamEndCapture in a
     * different thread.
     */
    cudaErrorStreamCaptureWrongThread     =    908,

    /**
     * This indicates that the wait operation has timed out.
     */
    cudaErrorTimeout                      =    909,

    /**
     * This error indicates that the graph update was not performed because it included 
     * changes which violated constraints specific to instantiated graph update.
     */
    cudaErrorGraphExecUpdateFailure       =    910,

    /**
     * This indicates that an async error has occurred in a device outside of CUDA.
     * If CUDA was waiting for an external device's signal before consuming shared data,
     * the external device signaled an error indicating that the data is not valid for
     * consumption. This leaves the process in an inconsistent state and any further CUDA
     * work will return the same error. To continue using CUDA, the process must be
     * terminated and relaunched.
     */
    cudaErrorExternalDevice               =    911,

    /**
     * This indicates that a kernel launch error has occurred due to cluster
     * misconfiguration.
     */
    cudaErrorInvalidClusterSize           =    912,

    /**
     * Indiciates a function handle is not loaded when calling an API that requires
     * a loaded function.
     */
    cudaErrorFunctionNotLoaded            =    913,

    /**
     * This error indicates one or more resources passed in are not valid resource
     * types for the operation.
     */
    cudaErrorInvalidResourceType          =    914,

    /**
     * This error indicates one or more resources are insufficient or non-applicable for
     * the operation.
     */
    cudaErrorInvalidResourceConfiguration =    915,

    /**
     * This error indicates that the requested operation is not permitted because the
     * stream is in a detached state. This can occur if the green context associated
     * with the stream has been destroyed, limiting the stream's operational capabilities.
     */
    cudaErrorStreamDetached               =    917,

    /**
     * This indicates that an unknown internal error has occurred.
     */
    cudaErrorUnknown                      =    999

    /**
     * Any unhandled CUDA driver error is added to this value and returned via
     * the runtime. Production releases of CUDA should not return such errors.
     * \deprecated
     * This error return is deprecated as of CUDA 4.1.
     */
    , cudaErrorApiFailureBase               =  10000
};

/**
 * Channel format kind
 */
enum __device_builtin__ cudaChannelFormatKind
{
    cudaChannelFormatKindSigned                         =   0,      /**< Signed channel format */
    cudaChannelFormatKindUnsigned                       =   1,      /**< Unsigned channel format */
    cudaChannelFormatKindFloat                          =   2,      /**< Float channel format */
    cudaChannelFormatKindNone                           =   3,      /**< No channel format */
    cudaChannelFormatKindNV12                           =   4,      /**< Unsigned 8-bit integers, planar 4:2:0 YUV format */
    cudaChannelFormatKindUnsignedNormalized8X1          =   5,      /**< 1 channel unsigned 8-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized8X2          =   6,      /**< 2 channel unsigned 8-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized8X4          =   7,      /**< 4 channel unsigned 8-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized16X1         =   8,      /**< 1 channel unsigned 16-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized16X2         =   9,      /**< 2 channel unsigned 16-bit normalized integer */
    cudaChannelFormatKindUnsignedNormalized16X4         =   10,     /**< 4 channel unsigned 16-bit normalized integer */
    cudaChannelFormatKindSignedNormalized8X1            =   11,     /**< 1 channel signed 8-bit normalized integer */
    cudaChannelFormatKindSignedNormalized8X2            =   12,     /**< 2 channel signed 8-bit normalized integer */
    cudaChannelFormatKindSignedNormalized8X4            =   13,     /**< 4 channel signed 8-bit normalized integer */
    cudaChannelFormatKindSignedNormalized16X1           =   14,     /**< 1 channel signed 16-bit normalized integer */
    cudaChannelFormatKindSignedNormalized16X2           =   15,     /**< 2 channel signed 16-bit normalized integer */
    cudaChannelFormatKindSignedNormalized16X4           =   16,     /**< 4 channel signed 16-bit normalized integer */
    cudaChannelFormatKindUnsignedBlockCompressed1       =   17,     /**< 4 channel unsigned normalized block-compressed (BC1 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB   =   18,     /**< 4 channel unsigned normalized block-compressed (BC1 compression) format with sRGB encoding*/
    cudaChannelFormatKindUnsignedBlockCompressed2       =   19,     /**< 4 channel unsigned normalized block-compressed (BC2 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB   =   20,     /**< 4 channel unsigned normalized block-compressed (BC2 compression) format with sRGB encoding */
    cudaChannelFormatKindUnsignedBlockCompressed3       =   21,     /**< 4 channel unsigned normalized block-compressed (BC3 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB   =   22,     /**< 4 channel unsigned normalized block-compressed (BC3 compression) format with sRGB encoding */
    cudaChannelFormatKindUnsignedBlockCompressed4       =   23,     /**< 1 channel unsigned normalized block-compressed (BC4 compression) format */
    cudaChannelFormatKindSignedBlockCompressed4         =   24,     /**< 1 channel signed normalized block-compressed (BC4 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed5       =   25,     /**< 2 channel unsigned normalized block-compressed (BC5 compression) format */
    cudaChannelFormatKindSignedBlockCompressed5         =   26,     /**< 2 channel signed normalized block-compressed (BC5 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed6H      =   27,     /**< 3 channel unsigned half-float block-compressed (BC6H compression) format */
    cudaChannelFormatKindSignedBlockCompressed6H        =   28,     /**< 3 channel signed half-float block-compressed (BC6H compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed7       =   29,     /**< 4 channel unsigned normalized block-compressed (BC7 compression) format */
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB   =   30,     /**< 4 channel unsigned normalized block-compressed (BC7 compression) format with sRGB encoding */
    cudaChannelFormatKindUnsignedNormalized1010102      =   31      /**< 4 channel unsigned normalized (10-bit, 10-bit, 10-bit, 2-bit) format */

};

/**
 * An opaque descriptor handle. The descriptor encapsulates multiple created and configured resources.
 * Created via ::cudaDeviceResourceGenerateDesc
 */
typedef __device_builtin__ struct CUdevResourceDesc_st *cudaDevResourceDesc_t;

/**
 * An opaque handle to a CUDA execution context. It represents an execution context created via CUDA Runtime APIs such as ::cudaGreenCtxCreate.
 */
typedef __device_builtin__ struct cudaExecutionContext_st* cudaExecutionContext_t;

/**
 * CUDA Channel format descriptor
 */
struct __device_builtin__ cudaChannelFormatDesc
{
    int                        x; /**< x */
    int                        y; /**< y */
    int                        z; /**< z */
    int                        w; /**< w */
    enum cudaChannelFormatKind f; /**< Channel format kind */
};

/**
 * CUDA array
 */
typedef struct cudaArray *cudaArray_t;

/**
 * CUDA array (as source copy argument)
 */
typedef const struct cudaArray *cudaArray_const_t;

struct cudaArray;

/**
 * CUDA mipmapped array
 */
typedef struct cudaMipmappedArray *cudaMipmappedArray_t;

/**
 * CUDA mipmapped array (as source argument)
 */
typedef const struct cudaMipmappedArray *cudaMipmappedArray_const_t;

struct cudaMipmappedArray;

/**
 * Indicates that the layered sparse CUDA array or CUDA mipmapped array has a single mip tail region for all layers
 */
#define cudaArraySparsePropertiesSingleMipTail   0x1

/**
 * Sparse CUDA array and CUDA mipmapped array properties
 */
struct __device_builtin__ cudaArraySparseProperties {
    struct {
        unsigned int width;             /**< Tile width in elements */
        unsigned int height;            /**< Tile height in elements */
        unsigned int depth;             /**< Tile depth in elements */
    } tileExtent;
    unsigned int miptailFirstLevel;     /**< First mip level at which the mip tail begins */   
    unsigned long long miptailSize;     /**< Total size of the mip tail. */
    unsigned int flags;                 /**< Flags will either be zero or ::cudaArraySparsePropertiesSingleMipTail */
    unsigned int reserved[4];
};

/**
 * CUDA array and CUDA mipmapped array memory requirements
 */
struct __device_builtin__ cudaArrayMemoryRequirements {
    size_t size;                    /**< Total size of the array. */
    size_t alignment;               /**< Alignment necessary for mapping the array. */
    unsigned int reserved[4];
};

/**
 * CUDA memory types
 */
enum __device_builtin__ cudaMemoryType
{
    cudaMemoryTypeUnregistered = 0, /**< Unregistered memory */
    cudaMemoryTypeHost         = 1, /**< Host memory */
    cudaMemoryTypeDevice       = 2, /**< Device memory */
    cudaMemoryTypeManaged      = 3  /**< Managed memory */
};

/**
 * CUDA memory copy types
 */
enum __device_builtin__ cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      /**< Host   -> Host */
    cudaMemcpyHostToDevice        =   1,      /**< Host   -> Device */
    cudaMemcpyDeviceToHost        =   2,      /**< Device -> Host */
    cudaMemcpyDeviceToDevice      =   3,      /**< Device -> Device */
    cudaMemcpyDefault             =   4       /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};

/**
 * CUDA Pitched memory pointer
 *
 * \sa ::make_cudaPitchedPtr
 */
struct __device_builtin__ cudaPitchedPtr
{
    void   *ptr;      /**< Pointer to allocated memory */
    size_t  pitch;    /**< Pitch of allocated memory in bytes */
    size_t  xsize;    /**< Logical width of allocation in elements */
    size_t  ysize;    /**< Logical height of allocation in elements */
};

/**
 * CUDA extent
 *
 * \sa ::make_cudaExtent
 */
struct __device_builtin__ cudaExtent
{
    size_t width;     /**< Width in elements when referring to array memory, in bytes when referring to linear memory */
    size_t height;    /**< Height in elements */
    size_t depth;     /**< Depth in elements */
};

/**
 * CUDA 3D position
 *
 * \sa ::make_cudaPos
 */
struct __device_builtin__ cudaPos
{
    size_t x;     /**< x */
    size_t y;     /**< y */
    size_t z;     /**< z */
};

/**
 * CUDA 3D memory copying parameters
 */
struct __device_builtin__ cudaMemcpy3DParms
{
    cudaArray_t            srcArray;  /**< Source memory address */
    struct cudaPos         srcPos;    /**< Source position offset */
    struct cudaPitchedPtr  srcPtr;    /**< Pitched source memory address */
  
    cudaArray_t            dstArray;  /**< Destination memory address */
    struct cudaPos         dstPos;    /**< Destination position offset */
    struct cudaPitchedPtr  dstPtr;    /**< Pitched destination memory address */
  
    struct cudaExtent      extent;    /**< Requested memory copy size */
    enum cudaMemcpyKind    kind;      /**< Type of transfer */
};

/**
 * Memcpy node parameters
 */
struct __device_builtin__ cudaMemcpyNodeParams {
    int flags;                            /**< Must be zero */
    int reserved;                         /**< Must be zero */
    cudaExecutionContext_t ctx;           /**< Context in which to run the memcpy. If NULL will try to use the current context. */
    struct cudaMemcpy3DParms copyParams;  /**< Parameters for the memory copy */
};

/**
 * CUDA 3D cross-device memory copying parameters
 */
struct __device_builtin__ cudaMemcpy3DPeerParms
{
    cudaArray_t            srcArray;  /**< Source memory address */
    struct cudaPos         srcPos;    /**< Source position offset */
    struct cudaPitchedPtr  srcPtr;    /**< Pitched source memory address */
    int                    srcDevice; /**< Source device */
  
    cudaArray_t            dstArray;  /**< Destination memory address */
    struct cudaPos         dstPos;    /**< Destination position offset */
    struct cudaPitchedPtr  dstPtr;    /**< Pitched destination memory address */
    int                    dstDevice; /**< Destination device */
  
    struct cudaExtent      extent;    /**< Requested memory copy size */
};

/**
 * CUDA Memset node parameters
 */
struct __device_builtin__  cudaMemsetParams {
    void *dst;                              /**< Destination device pointer */
    size_t pitch;                           /**< Pitch of destination device pointer. Unused if height is 1 */
    unsigned int value;                     /**< Value to be set */
    unsigned int elementSize;               /**< Size of each element in bytes. Must be 1, 2, or 4. */
    size_t width;                           /**< Width of the row in elements */
    size_t height;                          /**< Number of rows */
};

/**
 * CUDA Memset node parameters
 */
struct __device_builtin__  cudaMemsetParamsV2 {
    void *dst;                              /**< Destination device pointer */
    size_t pitch;                           /**< Pitch of destination device pointer. Unused if height is 1 */
    unsigned int value;                     /**< Value to be set */
    unsigned int elementSize;               /**< Size of each element in bytes. Must be 1, 2, or 4. */
    size_t width;                           /**< Width of the row in elements */
    size_t height;                          /**< Number of rows */
    cudaExecutionContext_t ctx;             /**< Context in which to run the memset. If NULL will try to use the current context. */
};

/**
 * Specifies performance hint with ::cudaAccessPolicyWindow for hitProp and missProp members.
 */
enum __device_builtin__  cudaAccessProperty {
    cudaAccessPropertyNormal = 0,       /**< Normal cache persistence. */
    cudaAccessPropertyStreaming = 1,    /**< Streaming access is less likely to persit from cache. */
    cudaAccessPropertyPersisting = 2    /**< Persisting access is more likely to persist in cache.*/
};

/**
 * Specifies an access policy for a window, a contiguous extent of memory
 * beginning at base_ptr and ending at base_ptr + num_bytes.
 * Partition into many segments and assign segments such that.
 * sum of "hit segments" / window == approx. ratio.
 * sum of "miss segments" / window == approx 1-ratio.
 * Segments and ratio specifications are fitted to the capabilities of
 * the architecture.
 * Accesses in a hit segment apply the hitProp access policy.
 * Accesses in a miss segment apply the missProp access policy.
 */
struct __device_builtin__ cudaAccessPolicyWindow {
    void *base_ptr;                     /**< Starting address of the access policy window. CUDA driver may align it. */
    size_t num_bytes;                   /**< Size in bytes of the window policy. CUDA driver may restrict the maximum size and alignment. */
    float hitRatio;                     /**< hitRatio specifies percentage of lines assigned hitProp, rest are assigned missProp. */
    enum cudaAccessProperty hitProp;    /**< ::CUaccessProperty set for hit. */
    enum cudaAccessProperty missProp;   /**< ::CUaccessProperty set for miss. Must be either NORMAL or STREAMING. */
};

#ifdef _WIN32
#define CUDART_CB __stdcall
#else
#define CUDART_CB
#endif

/**
 * CUDA host function
 * \param userData Argument value passed to the function
 */
typedef void (CUDART_CB *cudaHostFn_t)(void *userData);

/**
 * CUDA host node parameters
 */
struct __device_builtin__ cudaHostNodeParams {
    cudaHostFn_t fn;    /**< The function to call when the node executes */
    void* userData; /**< Argument to pass to the function */
};

/**
 * CUDA host node parameters
 */
struct __device_builtin__ cudaHostNodeParamsV2 {
    cudaHostFn_t fn;    /**< The function to call when the node executes */
    void* userData; /**< Argument to pass to the function */
};

/**
 * Possible stream capture statuses returned by ::cudaStreamIsCapturing
 */
enum __device_builtin__ cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone        = 0, /**< Stream is not capturing */
    cudaStreamCaptureStatusActive      = 1, /**< Stream is actively capturing */
    cudaStreamCaptureStatusInvalidated = 2  /**< Stream is part of a capture sequence that
                                                   has been invalidated, but not terminated */
};

/**
 * Possible modes for stream capture thread interactions. For more details see
 * ::cudaStreamBeginCapture and ::cudaThreadExchangeStreamCaptureMode
 */
enum __device_builtin__ cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal      = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed     = 2
};

enum __device_builtin__ cudaSynchronizationPolicy {
    cudaSyncPolicyAuto = 1,
    cudaSyncPolicySpin = 2,
    cudaSyncPolicyYield = 3,
    cudaSyncPolicyBlockingSync = 4
};

/**
 * Cluster scheduling policies. These may be passed to ::cudaFuncSetAttribute
 */
enum __device_builtin__ cudaClusterSchedulingPolicy {
    cudaClusterSchedulingPolicyDefault       = 0, /**< the default policy */
    cudaClusterSchedulingPolicySpread        = 1, /**< spread the blocks within a cluster to the SMs */
    cudaClusterSchedulingPolicyLoadBalancing = 2  /**< allow the hardware to load-balance the blocks in a cluster to the SMs */
};

/**
 * Flags for ::cudaStreamUpdateCaptureDependencies
 */
enum __device_builtin__ cudaStreamUpdateCaptureDependenciesFlags {
    cudaStreamAddCaptureDependencies = 0x0, /**< Add new nodes to the dependency set */
    cudaStreamSetCaptureDependencies = 0x1  /**< Replace the dependency set with the new nodes */
};

/**
 * Flags for user objects for graphs
 */
enum __device_builtin__ cudaUserObjectFlags {
    cudaUserObjectNoDestructorSync = 0x1  /**< Indicates the destructor execution is not synchronized by any CUDA handle. */
};

/**
 * Flags for retaining user object references for graphs
 */
enum __device_builtin__ cudaUserObjectRetainFlags {
    cudaGraphUserObjectMove = 0x1  /**< Transfer references from the caller rather than creating new references. */
};

/**
 * CUDA graphics interop resource
 */
struct cudaGraphicsResource;

/**
 * CUDA graphics interop register flags
 */
enum __device_builtin__ cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone             = 0,  /**< Default */
    cudaGraphicsRegisterFlagsReadOnly         = 1,  /**< CUDA will not write to this resource */ 
    cudaGraphicsRegisterFlagsWriteDiscard     = 2,  /**< CUDA will only write to and will not read from this resource */
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,  /**< CUDA will bind this resource to a surface reference */
    cudaGraphicsRegisterFlagsTextureGather    = 8   /**< CUDA will perform texture gather operations on this resource */
};

/**
 * CUDA graphics interop map flags
 */
enum __device_builtin__ cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone         = 0,  /**< Default; Assume resource can be read/written */
    cudaGraphicsMapFlagsReadOnly     = 1,  /**< CUDA will not write to this resource */
    cudaGraphicsMapFlagsWriteDiscard = 2   /**< CUDA will only write to and will not read from this resource */
};

/**
 * CUDA graphics interop array indices for cube maps
 */
enum __device_builtin__ cudaGraphicsCubeFace 
{
    cudaGraphicsCubeFacePositiveX = 0x00, /**< Positive X face of cubemap */
    cudaGraphicsCubeFaceNegativeX = 0x01, /**< Negative X face of cubemap */
    cudaGraphicsCubeFacePositiveY = 0x02, /**< Positive Y face of cubemap */
    cudaGraphicsCubeFaceNegativeY = 0x03, /**< Negative Y face of cubemap */
    cudaGraphicsCubeFacePositiveZ = 0x04, /**< Positive Z face of cubemap */
    cudaGraphicsCubeFaceNegativeZ = 0x05  /**< Negative Z face of cubemap */
};

/**
 * CUDA resource types
 */
enum __device_builtin__ cudaResourceType
{
    cudaResourceTypeArray          = 0x00, /**< Array resource */
    cudaResourceTypeMipmappedArray = 0x01, /**< Mipmapped array resource */
    cudaResourceTypeLinear         = 0x02, /**< Linear resource */
    cudaResourceTypePitch2D        = 0x03  /**< Pitch 2D resource */
};

/**
 * CUDA texture resource view formats
 */
enum __device_builtin__ cudaResourceViewFormat
{
    cudaResViewFormatNone                      = 0x00, /**< No resource view format (use underlying resource format) */
    cudaResViewFormatUnsignedChar1             = 0x01, /**< 1 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar2             = 0x02, /**< 2 channel unsigned 8-bit integers */
    cudaResViewFormatUnsignedChar4             = 0x03, /**< 4 channel unsigned 8-bit integers */
    cudaResViewFormatSignedChar1               = 0x04, /**< 1 channel signed 8-bit integers */
    cudaResViewFormatSignedChar2               = 0x05, /**< 2 channel signed 8-bit integers */
    cudaResViewFormatSignedChar4               = 0x06, /**< 4 channel signed 8-bit integers */
    cudaResViewFormatUnsignedShort1            = 0x07, /**< 1 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort2            = 0x08, /**< 2 channel unsigned 16-bit integers */
    cudaResViewFormatUnsignedShort4            = 0x09, /**< 4 channel unsigned 16-bit integers */
    cudaResViewFormatSignedShort1              = 0x0a, /**< 1 channel signed 16-bit integers */
    cudaResViewFormatSignedShort2              = 0x0b, /**< 2 channel signed 16-bit integers */
    cudaResViewFormatSignedShort4              = 0x0c, /**< 4 channel signed 16-bit integers */
    cudaResViewFormatUnsignedInt1              = 0x0d, /**< 1 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt2              = 0x0e, /**< 2 channel unsigned 32-bit integers */
    cudaResViewFormatUnsignedInt4              = 0x0f, /**< 4 channel unsigned 32-bit integers */
    cudaResViewFormatSignedInt1                = 0x10, /**< 1 channel signed 32-bit integers */
    cudaResViewFormatSignedInt2                = 0x11, /**< 2 channel signed 32-bit integers */
    cudaResViewFormatSignedInt4                = 0x12, /**< 4 channel signed 32-bit integers */
    cudaResViewFormatHalf1                     = 0x13, /**< 1 channel 16-bit floating point */
    cudaResViewFormatHalf2                     = 0x14, /**< 2 channel 16-bit floating point */
    cudaResViewFormatHalf4                     = 0x15, /**< 4 channel 16-bit floating point */
    cudaResViewFormatFloat1                    = 0x16, /**< 1 channel 32-bit floating point */
    cudaResViewFormatFloat2                    = 0x17, /**< 2 channel 32-bit floating point */
    cudaResViewFormatFloat4                    = 0x18, /**< 4 channel 32-bit floating point */
    cudaResViewFormatUnsignedBlockCompressed1  = 0x19, /**< Block compressed 1 */
    cudaResViewFormatUnsignedBlockCompressed2  = 0x1a, /**< Block compressed 2 */
    cudaResViewFormatUnsignedBlockCompressed3  = 0x1b, /**< Block compressed 3 */
    cudaResViewFormatUnsignedBlockCompressed4  = 0x1c, /**< Block compressed 4 unsigned */
    cudaResViewFormatSignedBlockCompressed4    = 0x1d, /**< Block compressed 4 signed */
    cudaResViewFormatUnsignedBlockCompressed5  = 0x1e, /**< Block compressed 5 unsigned */
    cudaResViewFormatSignedBlockCompressed5    = 0x1f, /**< Block compressed 5 signed */
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20, /**< Block compressed 6 unsigned half-float */
    cudaResViewFormatSignedBlockCompressed6H   = 0x21, /**< Block compressed 6 signed half-float */
    cudaResViewFormatUnsignedBlockCompressed7  = 0x22  /**< Block compressed 7 */
};

/**
 * CUDA resource descriptor
 */
struct __device_builtin__ cudaResourceDesc {
    enum cudaResourceType resType;             /**< Resource type */
    
    union {
        struct {
            cudaArray_t array;                 /**< CUDA array */
        } array;
        struct {
            cudaMipmappedArray_t mipmap;       /**< CUDA mipmapped array */
        } mipmap;
        struct {
            void *devPtr;                      /**< Device pointer */
            struct cudaChannelFormatDesc desc; /**< Channel descriptor */
            size_t sizeInBytes;                /**< Size in bytes */
        } linear;
        struct {
            void *devPtr;                      /**< Device pointer */
            struct cudaChannelFormatDesc desc; /**< Channel descriptor */
            size_t width;                      /**< Width of the array in elements */
            size_t height;                     /**< Height of the array in elements */
            size_t pitchInBytes;               /**< Pitch between two rows in bytes */
        } pitch2D;
        struct {
            int reserved[32];
        } reserved;
    } res;

    unsigned int flags;                        /**< Flags (must be zero) */
};

/**
 * CUDA resource view descriptor
 */
struct __device_builtin__ cudaResourceViewDesc
{
    enum cudaResourceViewFormat format;           /**< Resource view format */
    size_t                      width;            /**< Width of the resource view */
    size_t                      height;           /**< Height of the resource view */
    size_t                      depth;            /**< Depth of the resource view */
    unsigned int                firstMipmapLevel; /**< First defined mipmap level */
    unsigned int                lastMipmapLevel;  /**< Last defined mipmap level */
    unsigned int                firstLayer;       /**< First layer index */
    unsigned int                lastLayer;        /**< Last layer index */
    unsigned int                reserved[16];     /**< Must be zero */
};

/**
 * CUDA pointer attributes
 */
struct __device_builtin__ cudaPointerAttributes
{
    /**
     * The type of memory - ::cudaMemoryTypeUnregistered, ::cudaMemoryTypeHost,
     * ::cudaMemoryTypeDevice or ::cudaMemoryTypeManaged.
     */
    enum cudaMemoryType type;

    /** 
     * The device against which the memory was allocated or registered.
     * If the memory type is ::cudaMemoryTypeDevice then this identifies 
     * the device on which the memory referred physically resides.  If
     * the memory type is ::cudaMemoryTypeHost or::cudaMemoryTypeManaged then
     * this identifies the device which was current when the memory was allocated
     * or registered (and if that device is deinitialized then this allocation
     * will vanish with that device's state).
     */
    int device;

    /**
     * The address which may be dereferenced on the current device to access 
     * the memory or NULL if no such address exists.
     */
    void *devicePointer;

    /**
     * The address which may be dereferenced on the host to access the
     * memory or NULL if no such address exists.
     *
     * \note CUDA doesn't check if unregistered memory is allocated so this field
     * may contain invalid pointer if an invalid pointer has been passed to CUDA.
     */
    void *hostPointer;

    /**
     * Must be zero
     */
    long reserved[8];
};

/**
 * CUDA function attributes
 */
struct __device_builtin__ cudaFuncAttributes
{
   /**
    * The size in bytes of statically-allocated shared memory per block
    * required by this function. This does not include dynamically-allocated
    * shared memory requested by the user at runtime.
    */
   size_t sharedSizeBytes;

   /**
    * The size in bytes of user-allocated constant memory required by this
    * function.
    */
   size_t constSizeBytes;

   /**
    * The size in bytes of local memory used by each thread of this function.
    */
   size_t localSizeBytes;

   /**
    * The maximum number of threads per block, beyond which a launch of the
    * function would fail. This number depends on both the function and the
    * device on which the function is currently loaded.
    */
   int maxThreadsPerBlock;

   /**
    * The number of registers used by each thread of this function.
    */
   int numRegs;

   /**
    * The PTX virtual architecture version for which the function was
    * compiled. This value is the major PTX version * 10 + the minor PTX
    * version, so a PTX version 1.3 function would return the value 13.
    */
   int ptxVersion;

   /**
    * The binary architecture version for which the function was compiled.
    * This value is the major binary version * 10 + the minor binary version,
    * so a binary version 1.3 function would return the value 13.
    */
   int binaryVersion;

   /**
    * The attribute to indicate whether the function has been compiled with 
    * user specified option "-Xptxas --dlcm=ca" set.
    */
   int cacheModeCA;

   /**
    * The maximum size in bytes of dynamic shared memory per block for 
    * this function. Any launch must have a dynamic shared memory size
    * smaller than this value.
    */
   int maxDynamicSharedSizeBytes;

   /**
    * On devices where the L1 cache and shared memory use the same hardware resources, 
    * this sets the shared memory carveout preference, in percent of the maximum shared memory. 
    * Refer to ::cudaDevAttrMaxSharedMemoryPerMultiprocessor.
    * This is only a hint, and the driver can choose a different ratio if required to execute the function.
    * See ::cudaFuncSetAttribute
    */
   int preferredShmemCarveout;

   /**
    * If this attribute is set, the kernel must launch with a valid cluster dimension
    * specified.
    */
   int clusterDimMustBeSet;

   /**
    * The required cluster width/height/depth in blocks. The values must either
    * all be 0 or all be positive. The validity of the cluster dimensions is
    * otherwise checked at launch time.
    *
    * If the value is set during compile time, it cannot be set at runtime.
    * Setting it at runtime should return cudaErrorNotPermitted.
    * See ::cudaFuncSetAttribute
    */
   int requiredClusterWidth;
   int requiredClusterHeight;
   int requiredClusterDepth;

   /**
    * The block scheduling policy of a function.
    * See ::cudaFuncSetAttribute
    */
   int clusterSchedulingPolicyPreference;

   /**
    * Whether the function can be launched with non-portable cluster size. 1 is
    * allowed, 0 is disallowed. A non-portable cluster size may only function
    * on the specific SKUs the program is tested on. The launch might fail if
    * the program is run on a different hardware platform.
    *
    * CUDA API provides ::cudaOccupancyMaxActiveClusters to assist with checking
    * whether the desired size can be launched on the current device.
    *
    * Portable Cluster Size
    *
    * A portable cluster size is guaranteed to be functional on all compute
    * capabilities higher than the target compute capability. The portable
    * cluster size for sm_90 is 8 blocks per cluster. This value may increase
    * for future compute capabilities.
    *
    * The specific hardware unit may support higher cluster sizes that’s not
    * guaranteed to be portable.
    * See ::cudaFuncSetAttribute
    */
   int nonPortableClusterSizeAllowed;

   /**
    * Reserved for future use.
    */
   int reserved[16];
};

/**
 * CUDA function attributes that can be set using ::cudaFuncSetAttribute
 */
enum __device_builtin__ cudaFuncAttribute
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8, /**< Maximum dynamic shared memory size */
    cudaFuncAttributePreferredSharedMemoryCarveout = 9, /**< Preferred shared memory-L1 cache split */
    cudaFuncAttributeClusterDimMustBeSet = 10, /**< Indicator to enforce valid cluster dimension specification on kernel launch */
    cudaFuncAttributeRequiredClusterWidth = 11, /**< Required cluster width */
    cudaFuncAttributeRequiredClusterHeight = 12, /**< Required cluster height */
    cudaFuncAttributeRequiredClusterDepth = 13, /**< Required cluster depth */
    cudaFuncAttributeNonPortableClusterSizeAllowed = 14, /**< Whether non-portable cluster scheduling policy is supported */
    cudaFuncAttributeClusterSchedulingPolicyPreference = 15, /**< Required cluster scheduling policy preference */
    cudaFuncAttributeMax
};

/**
 * CUDA function cache configurations
 */
enum __device_builtin__ cudaFuncCache
{
    cudaFuncCachePreferNone   = 0,    /**< Default function cache configuration, no preference */
    cudaFuncCachePreferShared = 1,    /**< Prefer larger shared memory and smaller L1 cache  */
    cudaFuncCachePreferL1     = 2,    /**< Prefer larger L1 cache and smaller shared memory */
    cudaFuncCachePreferEqual  = 3     /**< Prefer equal size L1 cache and shared memory */
};

/**
 * CUDA shared memory configuration
 * \deprecated
 */
enum __device_builtin__ cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};

/** 
 * Shared memory carveout configurations. These may be passed to cudaFuncSetAttribute
 */
enum __device_builtin__ cudaSharedCarveout {
    cudaSharedmemCarveoutDefault      = -1,  /**< No preference for shared memory or L1 (default) */
    cudaSharedmemCarveoutMaxShared    = 100, /**< Prefer maximum available shared memory, minimum L1 cache */
    cudaSharedmemCarveoutMaxL1        = 0    /**< Prefer maximum available L1 cache, minimum shared memory */
};

/**
 * CUDA device compute modes
 */
enum __device_builtin__ cudaComputeMode
{
    cudaComputeModeDefault          = 0,  /**< Default compute mode (Multiple threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusive        = 1,  /**< Compute-exclusive-thread mode (Only one thread in one process will be able to use ::cudaSetDevice() with this device) */
    cudaComputeModeProhibited       = 2,  /**< Compute-prohibited mode (No threads can use ::cudaSetDevice() with this device) */
    cudaComputeModeExclusiveProcess = 3   /**< Compute-exclusive-process mode (Many threads in one process will be able to use ::cudaSetDevice() with this device) */
};

/**
 * CUDA Limits
 */
enum __device_builtin__ cudaLimit
{
    cudaLimitStackSize                    = 0x00, /**< GPU thread stack size */
    cudaLimitPrintfFifoSize               = 0x01, /**< GPU printf FIFO size */
    cudaLimitMallocHeapSize               = 0x02, /**< GPU malloc heap size */
    cudaLimitDevRuntimeSyncDepth          = 0x03, /**< GPU device runtime synchronize depth */
    cudaLimitDevRuntimePendingLaunchCount = 0x04, /**< GPU device runtime pending launch count */
    cudaLimitMaxL2FetchGranularity        = 0x05, /**< A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint */
    cudaLimitPersistingL2CacheSize        = 0x06  /**< A size in bytes for L2 persisting lines cache size */
};

/**
 * CUDA Memory Advise values
 */
enum __device_builtin__ cudaMemoryAdvise
{
    cudaMemAdviseSetReadMostly          = 1, /**< Data will mostly be read and only occassionally be written to */
    cudaMemAdviseUnsetReadMostly        = 2, /**< Undo the effect of ::cudaMemAdviseSetReadMostly */
    cudaMemAdviseSetPreferredLocation   = 3, /**< Set the preferred location for the data as the specified device */
    cudaMemAdviseUnsetPreferredLocation = 4, /**< Clear the preferred location for the data */
    cudaMemAdviseSetAccessedBy          = 5, /**< Data will be accessed by the specified device, so prevent page faults as much as possible */
    cudaMemAdviseUnsetAccessedBy        = 6  /**< Let the Unified Memory subsystem decide on the page faulting policy for the specified device */
};

/**
 * CUDA range attributes
 */
enum __device_builtin__ cudaMemRangeAttribute
{
    cudaMemRangeAttributeReadMostly               = 1, /**< Whether the range will mostly be read and only occassionally be written to */
    cudaMemRangeAttributePreferredLocation        = 2, /**< The preferred location of the range */
    cudaMemRangeAttributeAccessedBy               = 3, /**< Memory range has ::cudaMemAdviseSetAccessedBy set for specified device */
    cudaMemRangeAttributeLastPrefetchLocation     = 4, /**< The last location to which the range was prefetched */
    cudaMemRangeAttributePreferredLocationType    = 5, /**< The preferred location type of the range */
    cudaMemRangeAttributePreferredLocationId      = 6, /**< The preferred location id of the range */
    cudaMemRangeAttributeLastPrefetchLocationType = 7, /**< The last location type to which the range was prefetched */
    cudaMemRangeAttributeLastPrefetchLocationId   = 8  /**< The last location id to which the range was prefetched */
};

/**
 * CUDA GPUDirect RDMA flush writes APIs supported on the device
 */
enum __device_builtin__ cudaFlushGPUDirectRDMAWritesOptions {
    cudaFlushGPUDirectRDMAWritesOptionHost   = 1<<0, /**< ::cudaDeviceFlushGPUDirectRDMAWrites() and its CUDA Driver API counterpart are supported on the device. */
    cudaFlushGPUDirectRDMAWritesOptionMemOps = 1<<1  /**< The ::CU_STREAM_WAIT_VALUE_FLUSH flag and the ::CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES MemOp are supported on the CUDA device. */
};

/**
 * CUDA GPUDirect RDMA flush writes ordering features of the device
 */
enum __device_builtin__ cudaGPUDirectRDMAWritesOrdering {
    cudaGPUDirectRDMAWritesOrderingNone       = 0,   /**< The device does not natively support ordering of GPUDirect RDMA writes. ::cudaFlushGPUDirectRDMAWrites() can be leveraged if supported. */
    cudaGPUDirectRDMAWritesOrderingOwner      = 100, /**< Natively, the device can consistently consume GPUDirect RDMA writes, although other CUDA devices may not. */
    cudaGPUDirectRDMAWritesOrderingAllDevices = 200  /**< Any CUDA device in the system can consistently consume GPUDirect RDMA writes to this device. */
};

/**
 * CUDA GPUDirect RDMA flush writes scopes
 */
enum __device_builtin__ cudaFlushGPUDirectRDMAWritesScope {
    cudaFlushGPUDirectRDMAWritesToOwner      = 100, /**< Blocks until remote writes are visible to the CUDA device context owning the data. */
    cudaFlushGPUDirectRDMAWritesToAllDevices = 200  /**< Blocks until remote writes are visible to all CUDA device contexts. */
};

/**
 * CUDA GPUDirect RDMA flush writes targets
 */
enum __device_builtin__ cudaFlushGPUDirectRDMAWritesTarget {
    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice /**< Sets the target for ::cudaDeviceFlushGPUDirectRDMAWrites() to the currently active CUDA device context. */
};


/**
 * CUDA device attributes
 */
enum __device_builtin__ cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock             = 1,  /**< Maximum number of threads per block */
    cudaDevAttrMaxBlockDimX                   = 2,  /**< Maximum block dimension X */
    cudaDevAttrMaxBlockDimY                   = 3,  /**< Maximum block dimension Y */
    cudaDevAttrMaxBlockDimZ                   = 4,  /**< Maximum block dimension Z */
    cudaDevAttrMaxGridDimX                    = 5,  /**< Maximum grid dimension X */
    cudaDevAttrMaxGridDimY                    = 6,  /**< Maximum grid dimension Y */
    cudaDevAttrMaxGridDimZ                    = 7,  /**< Maximum grid dimension Z */
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,  /**< Maximum shared memory available per block in bytes */
    cudaDevAttrTotalConstantMemory            = 9,  /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
    cudaDevAttrWarpSize                       = 10, /**< Warp size in threads */
    cudaDevAttrMaxPitch                       = 11, /**< Maximum pitch in bytes allowed by memory copies */
    cudaDevAttrMaxRegistersPerBlock           = 12, /**< Maximum number of 32-bit registers available per block */
    cudaDevAttrClockRate                      = 13, /**< Peak clock frequency in kilohertz */
    cudaDevAttrTextureAlignment               = 14, /**< Alignment requirement for textures */
    cudaDevAttrGpuOverlap                     = 15, /**< Device can possibly copy memory and execute a kernel concurrently */
    cudaDevAttrMultiProcessorCount            = 16, /**< Number of multiprocessors on device */
    cudaDevAttrKernelExecTimeout              = 17, /**< Specifies whether there is a run time limit on kernels */
    cudaDevAttrIntegrated                     = 18, /**< Device is integrated with host memory */
    cudaDevAttrCanMapHostMemory               = 19, /**< Device can map host memory into CUDA address space */
    cudaDevAttrComputeMode                    = 20, /**< Compute mode (See ::cudaComputeMode for details) */
    cudaDevAttrMaxTexture1DWidth              = 21, /**< Maximum 1D texture width */
    cudaDevAttrMaxTexture2DWidth              = 22, /**< Maximum 2D texture width */
    cudaDevAttrMaxTexture2DHeight             = 23, /**< Maximum 2D texture height */
    cudaDevAttrMaxTexture3DWidth              = 24, /**< Maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeight             = 25, /**< Maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepth              = 26, /**< Maximum 3D texture depth */
    cudaDevAttrMaxTexture2DLayeredWidth       = 27, /**< Maximum 2D layered texture width */
    cudaDevAttrMaxTexture2DLayeredHeight      = 28, /**< Maximum 2D layered texture height */
    cudaDevAttrMaxTexture2DLayeredLayers      = 29, /**< Maximum layers in a 2D layered texture */
    cudaDevAttrSurfaceAlignment               = 30, /**< Alignment requirement for surfaces */
    cudaDevAttrConcurrentKernels              = 31, /**< Device can possibly execute multiple kernels concurrently */
    cudaDevAttrEccEnabled                     = 32, /**< Device has ECC support enabled */
    cudaDevAttrPciBusId                       = 33, /**< PCI bus ID of the device */
    cudaDevAttrPciDeviceId                    = 34, /**< PCI device ID of the device */
    cudaDevAttrTccDriver                      = 35, /**< Device is using TCC driver model */
    cudaDevAttrMemoryClockRate                = 36, /**< Peak memory clock frequency in kilohertz */
    cudaDevAttrGlobalMemoryBusWidth           = 37, /**< Global memory bus width in bits */
    cudaDevAttrL2CacheSize                    = 38, /**< Size of L2 cache in bytes */
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39, /**< Maximum resident threads per multiprocessor */
    cudaDevAttrAsyncEngineCount               = 40, /**< Number of asynchronous engines */
    cudaDevAttrUnifiedAddressing              = 41, /**< Device shares a unified address space with the host */    
    cudaDevAttrMaxTexture1DLayeredWidth       = 42, /**< Maximum 1D layered texture width */
    cudaDevAttrMaxTexture1DLayeredLayers      = 43, /**< Maximum layers in a 1D layered texture */
    cudaDevAttrMaxTexture2DGatherWidth        = 45, /**< Maximum 2D texture width if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture2DGatherHeight       = 46, /**< Maximum 2D texture height if cudaArrayTextureGather is set */
    cudaDevAttrMaxTexture3DWidthAlt           = 47, /**< Alternate maximum 3D texture width */
    cudaDevAttrMaxTexture3DHeightAlt          = 48, /**< Alternate maximum 3D texture height */
    cudaDevAttrMaxTexture3DDepthAlt           = 49, /**< Alternate maximum 3D texture depth */
    cudaDevAttrPciDomainId                    = 50, /**< PCI domain ID of the device */
    cudaDevAttrTexturePitchAlignment          = 51, /**< Pitch alignment requirement for textures */
    cudaDevAttrMaxTextureCubemapWidth         = 52, /**< Maximum cubemap texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53, /**< Maximum cubemap layered texture width/height */
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54, /**< Maximum layers in a cubemap layered texture */
    cudaDevAttrMaxSurface1DWidth              = 55, /**< Maximum 1D surface width */
    cudaDevAttrMaxSurface2DWidth              = 56, /**< Maximum 2D surface width */
    cudaDevAttrMaxSurface2DHeight             = 57, /**< Maximum 2D surface height */
    cudaDevAttrMaxSurface3DWidth              = 58, /**< Maximum 3D surface width */
    cudaDevAttrMaxSurface3DHeight             = 59, /**< Maximum 3D surface height */
    cudaDevAttrMaxSurface3DDepth              = 60, /**< Maximum 3D surface depth */
    cudaDevAttrMaxSurface1DLayeredWidth       = 61, /**< Maximum 1D layered surface width */
    cudaDevAttrMaxSurface1DLayeredLayers      = 62, /**< Maximum layers in a 1D layered surface */
    cudaDevAttrMaxSurface2DLayeredWidth       = 63, /**< Maximum 2D layered surface width */
    cudaDevAttrMaxSurface2DLayeredHeight      = 64, /**< Maximum 2D layered surface height */
    cudaDevAttrMaxSurface2DLayeredLayers      = 65, /**< Maximum layers in a 2D layered surface */
    cudaDevAttrMaxSurfaceCubemapWidth         = 66, /**< Maximum cubemap surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67, /**< Maximum cubemap layered surface width */
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68, /**< Maximum layers in a cubemap layered surface */
    cudaDevAttrMaxTexture1DLinearWidth        = 69, /**< Maximum 1D linear texture width */
    cudaDevAttrMaxTexture2DLinearWidth        = 70, /**< Maximum 2D linear texture width */
    cudaDevAttrMaxTexture2DLinearHeight       = 71, /**< Maximum 2D linear texture height */
    cudaDevAttrMaxTexture2DLinearPitch        = 72, /**< Maximum 2D linear texture pitch in bytes */
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73, /**< Maximum mipmapped 2D texture width */
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74, /**< Maximum mipmapped 2D texture height */
    cudaDevAttrComputeCapabilityMajor         = 75, /**< Major compute capability version number */ 
    cudaDevAttrComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77, /**< Maximum mipmapped 1D texture width */
    cudaDevAttrStreamPrioritiesSupported      = 78, /**< Device supports stream priorities */
    cudaDevAttrGlobalL1CacheSupported         = 79, /**< Device supports caching globals in L1 */
    cudaDevAttrLocalL1CacheSupported          = 80, /**< Device supports caching locals in L1 */
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81, /**< Maximum shared memory available per multiprocessor in bytes */
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82, /**< Maximum number of 32-bit registers available per multiprocessor */
    cudaDevAttrManagedMemory                  = 83, /**< Device can allocate managed memory on this system */
    cudaDevAttrIsMultiGpuBoard                = 84, /**< Device is on a multi-GPU board */
    cudaDevAttrMultiGpuBoardGroupID           = 85, /**< Unique identifier for a group of devices on the same multi-GPU board */
    cudaDevAttrHostNativeAtomicSupported      = 86, /**< Link between the device and the host supports native atomic operations */
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87, /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
    cudaDevAttrPageableMemoryAccess           = 88, /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    cudaDevAttrConcurrentManagedAccess        = 89, /**< Device can coherently access managed memory concurrently with the CPU */
    cudaDevAttrComputePreemptionSupported     = 90, /**< Device supports Compute Preemption */
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91, /**< Device can access host registered memory at the same virtual address as the CPU */
    cudaDevAttrReserved92                     = 92,
    cudaDevAttrReserved93                     = 93,
    cudaDevAttrReserved94                     = 94,
    cudaDevAttrCooperativeLaunch              = 95, /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel*/
    cudaDevAttrReserved96                     = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin   = 97, /**< The maximum optin shared memory per block. This value may vary by chip. See ::cudaFuncSetAttribute */
    cudaDevAttrCanFlushRemoteWrites           = 98, /**< Device supports flushing of outstanding remote writes. */
    cudaDevAttrHostRegisterSupported          = 99, /**< Device supports host memory registration via ::cudaHostRegister. */
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100, /**< Device accesses pageable memory via the host's page tables. */
    cudaDevAttrDirectManagedMemAccessFromHost = 101, /**< Host can directly access managed memory on the device without migration. */
    cudaDevAttrMaxBlocksPerMultiprocessor     = 106, /**< Maximum number of blocks per multiprocessor */
    cudaDevAttrMaxPersistingL2CacheSize       = 108, /**< Maximum L2 persisting lines capacity setting in bytes. */
    cudaDevAttrMaxAccessPolicyWindowSize      = 109, /**< Maximum value of cudaAccessPolicyWindow::num_bytes. */
    cudaDevAttrReservedSharedMemoryPerBlock   = 111, /**< Shared memory reserved by CUDA driver per block in bytes */
    cudaDevAttrSparseCudaArraySupported       = 112, /**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays */
    cudaDevAttrHostRegisterReadOnlySupported  = 113,  /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,  /**< External timeline semaphore interop is supported on the device */
    cudaDevAttrMemoryPoolsSupported           = 115, /**< Device supports using the ::cudaMallocAsync and ::cudaMemPool family of APIs */
    cudaDevAttrGPUDirectRDMASupported         = 116, /**< Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) */
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117, /**< The returned attribute shall be interpreted as a bitmask, where the individual bits are listed in the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    cudaDevAttrGPUDirectRDMAWritesOrdering    = 118, /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::cudaGPUDirectRDMAWritesOrdering for the numerical values returned here. */
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119, /**< Handle types supported with mempool based IPC */
    cudaDevAttrClusterLaunch                  = 120, /**< Indicates device supports cluster launch */
    cudaDevAttrDeferredMappingCudaArraySupported = 121, /**< Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    cudaDevAttrReserved122                    = 122,
    cudaDevAttrReserved123                    = 123,
    cudaDevAttrReserved124                    = 124,
    cudaDevAttrIpcEventSupport                = 125, /**< Device supports IPC Events. */ 
    cudaDevAttrMemSyncDomainCount             = 126, /**< Number of memory synchronization domains the device supports. */
    cudaDevAttrReserved127                    = 127,
    cudaDevAttrReserved128                    = 128,
    cudaDevAttrReserved129                    = 129,
    cudaDevAttrNumaConfig                     = 130, /**< NUMA configuration of a device: value is of type ::cudaDeviceNumaConfig enum */
    cudaDevAttrNumaId                         = 131, /**< NUMA node ID of the GPU memory */
    cudaDevAttrReserved132                    = 132,
    cudaDevAttrMpsEnabled                     = 133, /**< Contexts created on this device will be shared via MPS */
    cudaDevAttrHostNumaId                     = 134, /**< NUMA ID of the host node closest to the device or -1 when system does not support NUMA */
    cudaDevAttrD3D12CigSupported              = 135, /**< Device supports CIG with D3D12. */
    cudaDevAttrVulkanCigSupported             = 138, /**< Device supports CIG with Vulkan. */
    cudaDevAttrGpuPciDeviceId                 = 139, /**< The combined 16-bit PCI device ID and 16-bit PCI vendor ID. */
    cudaDevAttrGpuPciSubsystemId              = 140, /**< The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID. */
    cudaDevAttrReserved141                    = 141,
    cudaDevAttrHostNumaMemoryPoolsSupported   = 142, /**< Device supports HOST_NUMA location with the ::cudaMallocAsync and ::cudaMemPool family of APIs */
    cudaDevAttrHostNumaMultinodeIpcSupported  = 143, /**< Device supports HostNuma location IPC between nodes in a multi-node system. */
    cudaDevAttrHostMemoryPoolsSupported       = 144, /**< Device suports HOST location with the ::cuMemAllocAsync and ::cuMemPool family of APIs */
    cudaDevAttrReserved145                    = 145,
    cudaDevAttrOnlyPartialHostNativeAtomicSupported = 147,     /**< Link between the device and the host supports only some native atomic operations */

    cudaDevAttrMax
};

/**
 * CUDA memory pool attributes
 */
enum __device_builtin__ cudaMemPoolAttr
{
    /**
     * (value type = int)
     * Allow cuMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * Cuda events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     */
    cudaMemPoolReuseFollowEventDependencies   = 0x1,

    /**
     * (value type = int)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     */
    cudaMemPoolReuseAllowOpportunistic        = 0x2,

    /**
     * (value type = int)
     * Allow cuMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by cuFreeAsync (default enabled).
     */
    cudaMemPoolReuseAllowInternalDependencies = 0x3,


    /**
     * (value type = cuuint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     */
    cudaMemPoolAttrReleaseThreshold           = 0x4,

    /**
     * (value type = cuuint64_t)
     * Amount of backing memory currently allocated for the mempool.
     */
    cudaMemPoolAttrReservedMemCurrent         = 0x5,

    /**
     * (value type = cuuint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     */
    cudaMemPoolAttrReservedMemHigh            = 0x6,

    /**
     * (value type = cuuint64_t)
     * Amount of memory from the pool that is currently in use by the application.
     */
    cudaMemPoolAttrUsedMemCurrent             = 0x7,

    /**
     * (value type = cuuint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     */
    cudaMemPoolAttrUsedMemHigh                = 0x8
};

/**
 * Specifies the type of location 
 */
enum __device_builtin__ cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeNone = 0, /**< Location is unspecified. This is used when creating a managed memory pool to indicate no preferred location for the pool */
    cudaMemLocationTypeDevice = 1,  /**< Location is a device location, thus id is a device ordinal */
    cudaMemLocationTypeHost = 2     /**< Location is host, id is ignored */
    , cudaMemLocationTypeHostNuma = 3 /**< Location is a host NUMA node, thus id is a host NUMA node id */
    , cudaMemLocationTypeHostNumaCurrent = 4 /**< Location is the host NUMA node closest to the current thread's CPU, id is ignored */
};

/**
 * Specifies a memory location.
 *
 * To specify a gpu, set type = ::cudaMemLocationTypeDevice and set id = the gpu's device ordinal.
 * To specify a cpu NUMA node, set type = ::cudaMemLocationTypeHostNuma and set id = host NUMA node id.
 */
struct __device_builtin__ cudaMemLocation {
    enum cudaMemLocationType type;  /**< Specifies the location type, which modifies the meaning of id. */
    int id;                         /**< identifier for a given this location's ::CUmemLocationType. */
};

/**
 * Specifies the memory protection flags for mapping.
 */
enum __device_builtin__ cudaMemAccessFlags {
    cudaMemAccessFlagsProtNone      = 0,  /**< Default, make the address range not accessible */
    cudaMemAccessFlagsProtRead      = 1,  /**< Make the address range read accessible */
    cudaMemAccessFlagsProtReadWrite = 3   /**< Make the address range read-write accessible */
};

/**
 * Memory access descriptor
 */
struct __device_builtin__ cudaMemAccessDesc {
    struct cudaMemLocation  location; /**< Location on which the request is to change it's accessibility */
    enum cudaMemAccessFlags flags;    /**< ::CUmemProt accessibility flags to set on the request */
};

/**
 * Defines the allocation types available
 */
enum __device_builtin__ cudaMemAllocationType {
    cudaMemAllocationTypeInvalid = 0x0,
    /** This allocation type is 'pinned', i.e. cannot migrate from its current
      * location while the application is actively using it
      */
    cudaMemAllocationTypePinned  = 0x1,
    /** This allocation type is managed memory
      */
    cudaMemAllocationTypeManaged = 0x2,
    cudaMemAllocationTypeMax     = 0x7FFFFFFF 
};

/**
 * Flags for specifying particular handle types
 */
enum __device_builtin__ cudaMemAllocationHandleType {
    cudaMemHandleTypeNone                    = 0x0,  /**< Does not allow any export mechanism. > */
    cudaMemHandleTypePosixFileDescriptor     = 0x1,  /**< Allows a file descriptor to be used for exporting. Permitted only on POSIX systems. (int) */
    cudaMemHandleTypeWin32                   = 0x2,  /**< Allows a Win32 NT handle to be used for exporting. (HANDLE) */
    cudaMemHandleTypeWin32Kmt                = 0x4,   /**< Allows a Win32 KMT handle to be used for exporting. (D3DKMT_HANDLE) */
    cudaMemHandleTypeFabric                  = 0x8  /**< Allows a fabric handle to be used for exporting. (cudaMemFabricHandle_t) */
};

/**
 * This flag, if set, indicates that the memory will be used as a buffer for
 * hardware accelerated decompression.
 */
#define cudaMemPoolCreateUsageHwDecompress 0x2

/**
 * Specifies the properties of allocations made from the pool.
 */
struct __device_builtin__ cudaMemPoolProps {
    enum cudaMemAllocationType         allocType;   /**< Allocation type. Currently must be specified as cudaMemAllocationTypePinned */
    enum cudaMemAllocationHandleType   handleTypes; /**< Handle types that will be supported by allocations from the pool. */
    struct cudaMemLocation             location;    /**< Location allocations should reside. */
    /**
     * Windows-specific LPSECURITYATTRIBUTES required when
     * ::cudaMemHandleTypeWin32 is specified.  This security attribute defines
     * the scope of which exported allocations may be tranferred to other
     * processes.  In all other cases, this field is required to be zero.
     */
    void                              *win32SecurityAttributes;
    size_t                             maxSize;     /**< Maximum pool size. When set to 0, defaults to a system dependent value.*/
    unsigned short                     usage;        /**< Bitmask indicating intended usage for the pool. */
    unsigned char                      reserved[54]; /**< reserved for future use, must be 0 */
};

/**
 * Opaque data for exporting a pool allocation
 */
struct __device_builtin__ cudaMemPoolPtrExportData {
    unsigned char reserved[64];
};

/**
 * Memory allocation node parameters
 */
struct __device_builtin__ cudaMemAllocNodeParams {
    /**
    * in: location where the allocation should reside (specified in ::location).
    * ::handleTypes must be ::cudaMemHandleTypeNone. IPC is not supported.
    */
    struct cudaMemPoolProps         poolProps;       /**< in: array of memory access descriptors. Used to describe peer GPU access */
    const struct cudaMemAccessDesc *accessDescs;     /**< in: number of memory access descriptors.  Must not exceed the number of GPUs. */
    size_t                          accessDescCount; /**< in: Number of `accessDescs`s */
    size_t                          bytesize;        /**< in: size in bytes of the requested allocation */
    void                           *dptr;            /**< out: address of the allocation returned by CUDA */
};

/**
 * Memory allocation node parameters
 */
struct __device_builtin__ cudaMemAllocNodeParamsV2 {
    /**
    * in: location where the allocation should reside (specified in ::location).
    * ::handleTypes must be ::cudaMemHandleTypeNone. IPC is not supported.
    */
    struct cudaMemPoolProps         poolProps;       /**< in: array of memory access descriptors. Used to describe peer GPU access */
    const struct cudaMemAccessDesc *accessDescs;     /**< in: number of memory access descriptors.  Must not exceed the number of GPUs. */
    size_t                          accessDescCount; /**< in: Number of `accessDescs`s */
    size_t                          bytesize;        /**< in: size in bytes of the requested allocation */
    void                           *dptr;            /**< out: address of the allocation returned by CUDA */
};

/**
 * Memory free node parameters
 */
struct __device_builtin__ cudaMemFreeNodeParams {
    void *dptr; /**< in: the pointer to free */
};

/**
 * Graph memory attributes
 */
enum __device_builtin__ cudaGraphMemAttributeType {
    /**
     * (value type = cuuint64_t)
     * Amount of memory, in bytes, currently associated with graphs.
     */
    cudaGraphMemAttrUsedMemCurrent      = 0x0,

    /**
     * (value type = cuuint64_t)
     * High watermark of memory, in bytes, associated with graphs since the
     * last time it was reset.  High watermark can only be reset to zero.
     */
    cudaGraphMemAttrUsedMemHigh         = 0x1,

    /**
     * (value type = cuuint64_t)
     * Amount of memory, in bytes, currently allocated for use by
     * the CUDA graphs asynchronous allocator.
     */
    cudaGraphMemAttrReservedMemCurrent  = 0x2,

    /**
     * (value type = cuuint64_t)
     * High watermark of memory, in bytes, currently allocated for use by
     * the CUDA graphs asynchronous allocator.
     */
    cudaGraphMemAttrReservedMemHigh     = 0x3
};

/**
 * Flags to specify for copies within a batch. For more details see ::cudaMemcpyBatchAsync.
 */
enum __device_builtin__ cudaMemcpyFlags {
    cudaMemcpyFlagDefault                  = 0x0,

    /**
     * Hint to the driver to try and overlap the copy with compute work on the SMs.
     */
    cudaMemcpyFlagPreferOverlapWithCompute = 0x1
};

enum __device_builtin__ cudaMemcpySrcAccessOrder {
    /**
     * Default invalid.
     */
    cudaMemcpySrcAccessOrderInvalid       = 0x0,

    /**
     * Indicates that access to the source pointer must be in stream order.
     */
    cudaMemcpySrcAccessOrderStream        = 0x1,

    /**
     * Indicates that access to the source pointer can be out of stream order and all
     * accesses must be complete before the API call returns. This flag is suited for
     * ephemeral sources (ex., stack variables) when it's known that no prior operations
     * in the stream can be accessing the memory and also that the lifetime of the memory
     * is limited to the scope that the source variable was declared in. Specifying
     * this flag allows the driver to optimize the copy and removes the need for the user
     * to synchronize the stream after the API call.
     */
    cudaMemcpySrcAccessOrderDuringApiCall = 0x2,

    /**
     * Indicates that access to the source pointer can be out of stream order and the accesses
     * can happen even after the API call returns. This flag is suited for host pointers
     * allocated outside CUDA (ex., via malloc) when it's known that no prior operations
     * in the stream can be accessing the memory. Specifying this flag allows the driver
     * to optimize the copy on certain platforms.
     */
    cudaMemcpySrcAccessOrderAny           = 0x3,

    cudaMemcpySrcAccessOrderMax           = 0x7FFFFFFF
};

/**
 * Attributes specific to copies within a batch. For more details on usage see ::cudaMemcpyBatchAsync.
 */
struct __device_builtin__ cudaMemcpyAttributes {
    enum cudaMemcpySrcAccessOrder srcAccessOrder;  /**< Source access ordering to be observed for copies with this attribute. */
    struct cudaMemLocation srcLocHint;             /**< Hint location for the source operand. Ignored when the pointers are not managed memory or memory allocated outside CUDA. */
    struct cudaMemLocation dstLocHint;             /**< Hint location for the destination operand. Ignored when the pointers are not managed memory or memory allocated outside CUDA. */
    unsigned int flags;                            /**< Additional flags for copies with this attribute. See ::cudaMemcpyFlags. */
};

/**
 * These flags allow applications to convey the operand type for individual copies specified in ::cudaMemcpy3DBatchAsync.
 */
enum __device_builtin__ cudaMemcpy3DOperandType {
    cudaMemcpyOperandTypePointer = 0x1,            /**< Memcpy operand is a valid pointer. */
    cudaMemcpyOperandTypeArray = 0x2,              /**< Memcpy operand is a CUarray. */
    cudaMemcpyOperandTypeMax = 0x7FFFFFFF
};

/**
 * Struct representing offset into a ::cudaArray_t in elements
 */
struct __device_builtin__ cudaOffset3D {
    size_t x;
    size_t y;
    size_t z;
};

/**
 * Struct representing an operand for copy with ::cudaMemcpy3DBatchAsync
 */
struct __device_builtin__ cudaMemcpy3DOperand {
    enum cudaMemcpy3DOperandType type;
    union {
        /**
         * Struct representing an operand when ::cudaMemcpy3DOperand::type is ::cudaMemcpyOperandTypePointer
         */
        struct {
            void *ptr;
            size_t rowLength;                /**< Length of each row in elements. */ 
            size_t layerHeight;              /**< Height of each layer in elements. */ 
            struct cudaMemLocation locHint;  /**< Hint location for the operand. Ignored when the pointers are not managed memory or memory allocated outside CUDA. */
        } ptr;

        /**
         * Struct representing an operand when ::cudaMemcpy3DOperand::type is ::cudaMemcpyOperandTypeArray
         */
        struct {
            cudaArray_t array;
            struct cudaOffset3D offset;
        } array;
    } op;  
};

struct __device_builtin__ cudaMemcpy3DBatchOp {
    struct cudaMemcpy3DOperand src;                /**< Source memcpy operand. */
    struct cudaMemcpy3DOperand dst;                /**< Destination memcpy operand. */
    struct cudaExtent extent;                      /**< Extents of the memcpy between src and dst. The width, height and depth components must not be 0.*/
    enum cudaMemcpySrcAccessOrder srcAccessOrder;  /**< Source access ordering to be observed for copy from src to dst. */
    unsigned int flags;                            /**< Additional flags for copy from src to dst. See ::cudaMemcpyFlags. */
};

/**
 * CUDA device P2P attributes
 */

enum __device_builtin__ cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank              = 1, /**< A relative value indicating the performance of the link between two devices */
    cudaDevP2PAttrAccessSupported              = 2, /**< Peer access is enabled */
    cudaDevP2PAttrNativeAtomicSupported        = 3, /**< Native atomic operation over the link supported */
    cudaDevP2PAttrCudaArrayAccessSupported     = 4  /**< Accessing CUDA arrays over the link supported */
    ,
    cudaDevP2PAttrOnlyPartialNativeAtomicSupported = 5   /**< Only some CUDA-valid atomic operations over the link are supported. */

};

/**
 * CUDA-valid Atomic Operations
 */
enum __device_builtin__ cudaAtomicOperation {
    cudaAtomicOperationIntegerAdd          = 0,
    cudaAtomicOperationIntegerMin          = 1,
    cudaAtomicOperationIntegerMax          = 2,
    cudaAtomicOperationIntegerIncrement    = 3,
    cudaAtomicOperationIntegerDecrement    = 4,
    cudaAtomicOperationAnd                 = 5,
    cudaAtomicOperationOr                  = 6,
    cudaAtomicOperationXOR                 = 7,
    cudaAtomicOperationExchange            = 8,
    cudaAtomicOperationCAS                 = 9,
    cudaAtomicOperationFloatAdd            = 10,
    cudaAtomicOperationFloatMin            = 11,
    cudaAtomicOperationFloatMax            = 12,
};

/** 
 * CUDA-valid Atomic Operation capabilities
 */
enum __device_builtin__ cudaAtomicOperationCapability {
    cudaAtomicCapabilitySigned          = 1u<<0,
    cudaAtomicCapabilityUnsigned        = 1u<<1,
    cudaAtomicCapabilityReduction       = 1u<<2,
    cudaAtomicCapabilityScalar32        = 1u<<3,
    cudaAtomicCapabilityScalar64        = 1u<<4,
    cudaAtomicCapabilityScalar128       = 1u<<5,
    cudaAtomicCapabilityVector32x4      = 1u<<6,
};



/**
 * CUDA UUID types
 */
#ifndef CU_UUID_HAS_BEEN_DEFINED
#define CU_UUID_HAS_BEEN_DEFINED
struct __device_builtin__ CUuuid_st {     /**< CUDA definition of UUID */
    char bytes[16];
};
typedef __device_builtin__ struct CUuuid_st CUuuid;
#endif
typedef __device_builtin__ struct CUuuid_st cudaUUID_t;

/**
 * CUDA device properties
 */
struct __device_builtin__ cudaDeviceProp
{
    char         name[256];                  /**< ASCII string identifying device */
    cudaUUID_t   uuid;                       /**< 16-byte unique identifier */
    char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
    unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
    size_t       totalGlobalMem;             /**< Global memory available on device in bytes */
    size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
    int          regsPerBlock;               /**< 32-bit registers available per block */
    int          warpSize;                   /**< Warp size in threads */
    size_t       memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
    int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
    int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
    int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
    size_t       totalConstMem;              /**< Constant memory available on device in bytes */
    int          major;                      /**< Major compute capability */
    int          minor;                      /**< Minor compute capability */
    size_t       textureAlignment;           /**< Alignment requirement for textures */
    size_t       texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
    int          multiProcessorCount;        /**< Number of multiprocessors on device */
    int          integrated;                 /**< Device is integrated as opposed to discrete */
    int          canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int          maxTexture1D;               /**< Maximum 1D texture size */
    int          maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
    int          maxTexture2D[2];            /**< Maximum 2D texture dimensions */
    int          maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
    int          maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int          maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int          maxTexture3D[3];            /**< Maximum 3D texture dimensions */
    int          maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
    int          maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
    int          maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
    int          maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
    int          maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
    int          maxSurface1D;               /**< Maximum 1D surface size */
    int          maxSurface2D[2];            /**< Maximum 2D surface dimensions */
    int          maxSurface3D[3];            /**< Maximum 3D surface dimensions */
    int          maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
    int          maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
    int          maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
    int          maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
    size_t       surfaceAlignment;           /**< Alignment requirements for surfaces */
    int          concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
    int          ECCEnabled;                 /**< Device has ECC support enabled */
    int          pciBusID;                   /**< PCI bus ID of the device */
    int          pciDeviceID;                /**< PCI device ID of the device */
    int          pciDomainID;                /**< PCI domain ID of the device */
    int          tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int          asyncEngineCount;           /**< Number of asynchronous engines */
    int          unifiedAddressing;          /**< Device shares a unified address space with the host */
    int          memoryBusWidth;             /**< Global memory bus width in bits */
    int          l2CacheSize;                /**< Size of L2 cache in bytes */
    int          persistingL2CacheMaxSize;   /**< Device's maximum l2 persisting lines capacity setting in bytes */
    int          maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
    int          streamPrioritiesSupported;  /**< Device supports stream priorities */
    int          globalL1CacheSupported;     /**< Device supports caching globals in L1 */
    int          localL1CacheSupported;      /**< Device supports caching locals in L1 */
    size_t       sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
    int          regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
    int          managedMemory;              /**< Device supports allocating managed memory on this system */
    int          isMultiGpuBoard;            /**< Device is on a multi-GPU board */
    int          multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
    int          hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
    int          pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    int          concurrentManagedAccess;    /**< Device can coherently access managed memory concurrently with the CPU */
    int          computePreemptionSupported; /**< Device supports Compute Preemption */
    int          canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
    int          cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
    size_t       sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
    int          pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */
    int          directManagedMemAccessFromHost; /**< Host can directly access managed memory on the device without migration. */
    int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
    int          accessPolicyMaxWindowSize;  /**< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
    size_t       reservedSharedMemPerBlock;  /**< Shared memory reserved by CUDA driver per block in bytes */
    int          hostRegisterSupported;      /**< Device supports host memory registration via ::cudaHostRegister. */
    int          sparseCudaArraySupported;   /**< 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise */
    int          hostRegisterReadOnlySupported; /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
    int          timelineSemaphoreInteropSupported; /**< External timeline semaphore interop is supported on the device */
    int          memoryPoolsSupported;       /**< 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise */
    int          gpuDirectRDMASupported;     /**< 1 if the device supports GPUDirect RDMA APIs, 0 otherwise */
    unsigned int gpuDirectRDMAFlushWritesOptions; /**< Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    int          gpuDirectRDMAWritesOrdering;/**< See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values */
    unsigned int memoryPoolSupportedHandleTypes; /**< Bitmask of handle types supported with mempool-based IPC */
    int          deferredMappingCudaArraySupported; /**< 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    int          ipcEventSupported;          /**< Device supports IPC Events. */
    int          clusterLaunch;              /**< Indicates device supports cluster launch */
    int          unifiedFunctionPointers;    /**< Indicates device supports unified pointers */
    int          deviceNumaConfig;           /**< NUMA configuration of a device: value is of type ::cudaDeviceNumaConfig enum */
    int          deviceNumaId;               /**< NUMA node ID of the GPU memory */
    int          mpsEnabled;                 /**< Indicates if contexts created on this device will be shared via MPS */
    int          hostNumaId;                 /**< NUMA ID of the host node closest to the device or -1 when system does not support NUMA */
    unsigned int gpuPciDeviceID; /**< The combined 16-bit PCI device ID and 16-bit PCI vendor ID */
    unsigned int gpuPciSubsystemID; /**< The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID */
    int          hostNumaMultinodeIpcSupported; /**< 1 if the device supports HostNuma location IPC between nodes in a multi-node system. */
    int          reserved[56];               /**< Reserved for future use */
};

/**
 * CUDA IPC Handle Size
 */
#define CUDA_IPC_HANDLE_SIZE 64

/**
 * CUDA IPC event handle
 */
typedef __device_builtin__ struct __device_builtin__ cudaIpcEventHandle_st
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcEventHandle_t;

/**
 * CUDA IPC memory handle
 */
typedef __device_builtin__ struct __device_builtin__ cudaIpcMemHandle_st 
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaIpcMemHandle_t;

/*
 * CUDA Mem Fabric Handle
 */
typedef __device_builtin__ struct __device_builtin__ cudaMemFabricHandle_st 
{
    char reserved[CUDA_IPC_HANDLE_SIZE];
}cudaMemFabricHandle_t;

/**
 * External memory handle types
 */
enum __device_builtin__ cudaExternalMemoryHandleType {
    /**
     * Handle is an opaque file descriptor
     */
    cudaExternalMemoryHandleTypeOpaqueFd         = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    cudaExternalMemoryHandleTypeOpaqueWin32      = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt   = 3,
    /**
     * Handle is a D3D12 heap object
     */
    cudaExternalMemoryHandleTypeD3D12Heap        = 4,
    /**
     * Handle is a D3D12 committed resource
     */
    cudaExternalMemoryHandleTypeD3D12Resource    = 5,
    /**
    *  Handle is a shared NT handle to a D3D11 resource
    */
    cudaExternalMemoryHandleTypeD3D11Resource    = 6,
    /**
    *  Handle is a globally shared handle to a D3D11 resource
    */
    cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
    /**
    *  Handle is an NvSciBuf object
    */
    cudaExternalMemoryHandleTypeNvSciBuf         = 8
};

/**
 * Indicates that the external memory object is a dedicated resource
 */
#define cudaExternalMemoryDedicated   0x1

/** When the /p flags parameter of ::cudaExternalSemaphoreSignalParams
 * contains this flag, it indicates that signaling an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::cudaExternalMemoryHandleTypeNvSciBuf,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same NvSciBuf memory objects.
 */
#define cudaExternalSemaphoreSignalSkipNvSciBufMemSync     0x01

/** When the /p flags parameter of ::cudaExternalSemaphoreWaitParams
 * contains this flag, it indicates that waiting an external semaphore object
 * should skip performing appropriate memory synchronization operations over all
 * the external memory objects that are imported as ::cudaExternalMemoryHandleTypeNvSciBuf,
 * which otherwise are performed by default to ensure data coherency with other
 * importers of the same NvSciBuf memory objects.
 */
#define cudaExternalSemaphoreWaitSkipNvSciBufMemSync       0x02

/**
 * When /p flags of ::cudaDeviceGetNvSciSyncAttributes is set to this,
 * it indicates that application need signaler specific NvSciSyncAttr
 * to be filled by ::cudaDeviceGetNvSciSyncAttributes.
 */
#define cudaNvSciSyncAttrSignal       0x1

/**
 * When /p flags of ::cudaDeviceGetNvSciSyncAttributes is set to this,
 * it indicates that application need waiter specific NvSciSyncAttr
 * to be filled by ::cudaDeviceGetNvSciSyncAttributes.
 */
#define cudaNvSciSyncAttrWait         0x2

/**
 * External memory handle descriptor
 */
struct __device_builtin__ cudaExternalMemoryHandleDesc {
    /**
     * Type of the handle
     */
    enum  cudaExternalMemoryHandleType type;
    union {
        /**
         * File descriptor referencing the memory object. Valid
         * when type is
         * ::cudaExternalMemoryHandleTypeOpaqueFd
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::cudaExternalMemoryHandleTypeOpaqueWin32
         * - ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt
         * - ::cudaExternalMemoryHandleTypeD3D12Heap 
         * - ::cudaExternalMemoryHandleTypeD3D12Resource
		 * - ::cudaExternalMemoryHandleTypeD3D11Resource
		 * - ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following: 
         * ::cudaExternalMemoryHandleTypeOpaqueWin32Kmt
         * ::cudaExternalMemoryHandleTypeD3D11ResourceKmt
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid memory object.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * A handle representing NvSciBuf Object. Valid when type
         * is ::cudaExternalMemoryHandleTypeNvSciBuf
         */
        const void *nvSciBufObject;
    } handle;
    /**
     * Size of the memory allocation
     */
    unsigned long long size;
    /**
     * Flags must either be zero or ::cudaExternalMemoryDedicated
     */
    unsigned int flags;
    /**
     * Must be zero
     */
    unsigned int reserved[16];
};

/**
 * External memory buffer descriptor
 */
struct __device_builtin__ cudaExternalMemoryBufferDesc {
    /**
     * Offset into the memory object where the buffer's base is
     */
    unsigned long long offset;
    /**
     * Size of the buffer
     */
    unsigned long long size;
    /**
     * Flags reserved for future use. Must be zero.
     */
    unsigned int flags;
    /**
     * Must be zero
     */
    unsigned int reserved[16];
};
 
/**
 * External memory mipmap descriptor
 */
struct __device_builtin__ cudaExternalMemoryMipmappedArrayDesc {
    /**
     * Offset into the memory object where the base level of the
     * mipmap chain is.
     */
    unsigned long long offset;
    /**
     * Format of base level of the mipmap chain
     */
    struct cudaChannelFormatDesc formatDesc;
    /**
     * Dimensions of base level of the mipmap chain
     */
    struct cudaExtent extent;
    /**
     * Flags associated with CUDA mipmapped arrays.
     * See ::cudaMallocMipmappedArray
     */
    unsigned int flags;
    /**
     * Total number of levels in the mipmap chain
     */
    unsigned int numLevels;
    /**
     * Must be zero
     */
    unsigned int reserved[16];
};
 
/**
 * External semaphore handle types
 */
enum __device_builtin__ cudaExternalSemaphoreHandleType {
    /**
     * Handle is an opaque file descriptor
     */
    cudaExternalSemaphoreHandleTypeOpaqueFd       = 1,
    /**
     * Handle is an opaque shared NT handle
     */
    cudaExternalSemaphoreHandleTypeOpaqueWin32    = 2,
    /**
     * Handle is an opaque, globally shared handle
     */
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
    /**
     * Handle is a shared NT handle referencing a D3D12 fence object
     */
    cudaExternalSemaphoreHandleTypeD3D12Fence     = 4,
    /**
     * Handle is a shared NT handle referencing a D3D11 fence object
     */
    cudaExternalSemaphoreHandleTypeD3D11Fence     = 5,
    /**
     * Opaque handle to NvSciSync Object
     */
     cudaExternalSemaphoreHandleTypeNvSciSync     = 6,
    /**
     * Handle is a shared NT handle referencing a D3D11 keyed mutex object
     */
    cudaExternalSemaphoreHandleTypeKeyedMutex     = 7,
    /**
     * Handle is a shared KMT handle referencing a D3D11 keyed mutex object
     */
    cudaExternalSemaphoreHandleTypeKeyedMutexKmt  = 8,
    /**
     * Handle is an opaque handle file descriptor referencing a timeline semaphore
     */
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd  = 9,
    /**
     * Handle is an opaque handle file descriptor referencing a timeline semaphore
     */
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32  = 10
};

/**
 * External semaphore handle descriptor
 */
struct __device_builtin__ cudaExternalSemaphoreHandleDesc {
    /**
     * Type of the handle
     */
    enum cudaExternalSemaphoreHandleType type;
    union {
        /**
         * File descriptor referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::cudaExternalSemaphoreHandleTypeOpaqueFd
         * - ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
         */
        int fd;
        /**
         * Win32 handle referencing the semaphore object. Valid when
         * type is one of the following:
         * - ::cudaExternalSemaphoreHandleTypeOpaqueWin32
         * - ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
         * - ::cudaExternalSemaphoreHandleTypeD3D12Fence
         * - ::cudaExternalSemaphoreHandleTypeD3D11Fence
         * - ::cudaExternalSemaphoreHandleTypeKeyedMutex
         * - ::cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
         * Exactly one of 'handle' and 'name' must be non-NULL. If
         * type is one of the following:
         * ::cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt
         * ::cudaExternalSemaphoreHandleTypeKeyedMutexKmt
         * then 'name' must be NULL.
         */
        struct {
            /**
             * Valid NT handle. Must be NULL if 'name' is non-NULL
             */
            void *handle;
            /**
             * Name of a valid synchronization primitive.
             * Must be NULL if 'handle' is non-NULL.
             */
            const void *name;
        } win32;
        /**
         * Valid NvSciSyncObj. Must be non NULL
         */
        const void* nvSciSyncObj;
    } handle;
    /**
     * Flags reserved for the future. Must be zero.
     */
    unsigned int flags;
    /**
     * Must be zero
     */
    unsigned int reserved[16];
};


/**
 * External semaphore signal parameters, compatible with driver type
 */
struct __device_builtin__ cudaExternalSemaphoreSignalParams{
    struct {
        /**
         * Parameters for fence objects
         */
        struct {
            /**
             * Value of fence to be signaled
             */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
             * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /*
             * Value of key to release the mutex with
             */
            unsigned long long key;
        } keyedMutex;
        unsigned int reserved[12];
    } params;
    /**
     * Only when ::cudaExternalSemaphoreSignalParams is used to
     * signal a ::cudaExternalSemaphore_t of type
     * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while signaling the ::cudaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
    unsigned int reserved[16];
};

/**
 * External semaphore wait parameters, compatible with driver type
 */
struct __device_builtin__ cudaExternalSemaphoreWaitParams {
    struct {
        /**
        * Parameters for fence objects
        */
        struct {
            /**
            * Value of fence to be waited on
            */
            unsigned long long value;
        } fence;
        union {
            /**
             * Pointer to NvSciSyncFence. Valid if ::cudaExternalSemaphoreHandleType
             * is of type ::cudaExternalSemaphoreHandleTypeNvSciSync.
             */
            void *fence;
            unsigned long long reserved;
        } nvSciSync;
        /**
         * Parameters for keyed mutex objects
         */
        struct {
            /**
             * Value of key to acquire the mutex with
             */
            unsigned long long key;
            /**
             * Timeout in milliseconds to wait to acquire the mutex
             */
            unsigned int timeoutMs;
        } keyedMutex;
        unsigned int reserved[10];
    } params;
    /**
     * Only when ::cudaExternalSemaphoreSignalParams is used to
     * signal a ::cudaExternalSemaphore_t of type
     * ::cudaExternalSemaphoreHandleTypeNvSciSync, the valid flag is 
     * ::cudaExternalSemaphoreSignalSkipNvSciBufMemSync: which indicates
     * that while waiting for the ::cudaExternalSemaphore_t, no memory
     * synchronization operations should be performed for any external memory
     * object imported as ::cudaExternalMemoryHandleTypeNvSciBuf.
     * For all other types of ::cudaExternalSemaphore_t, flags must be zero.
     */
    unsigned int flags;
    unsigned int reserved[16];
};

#define RESOURCE_ABI_BYTES 40

enum __device_builtin__ cudaDevSmResourceGroup_flags {
    cudaDevSmResourceGroupDefault = 0,
    cudaDevSmResourceGroupBackfill = 0x1
};

enum __device_builtin__ cudaDevSmResourceSplitByCount_flags {
    cudaDevSmResourceSplitIgnoreSmCoscheduling = 0x1,
    cudaDevSmResourceSplitMaxPotentialClusterSize = 0x2
};

/**
 * Type of resource
 */
enum __device_builtin__ cudaDevResourceType {
    cudaDevResourceTypeInvalid    = 0,
    cudaDevResourceTypeSm         = 1, /**< Streaming multiprocessors related information */
    cudaDevResourceTypeWorkqueueConfig = 1000, /**< Workqueue configuration related information */
    cudaDevResourceTypeWorkqueue = 10000, /**< Pre-existing workqueue related information */
};

/**
 * Data for SM-related resources
 * All parameters in this structure are OUTPUT only. Do not write to any of the fields in this structure.
 */
struct __device_builtin__ cudaDevSmResource {
    unsigned int smCount; /**< The amount of streaming multiprocessors available in this resource. */
    unsigned int minSmPartitionSize;  /**< The minimum number of streaming multiprocessors required to partition this resource. */
    unsigned int smCoscheduledAlignment; /**< The number of streaming multiprocessors in this resource that are guaranteed to
                                            be co-scheduled on the same GPU processing cluster. smCount will be a multiple of this value,
                                            unless the backfill flag is set. */
    unsigned int flags; /**< The flags set on this SM resource. For available flags see ::cudaDevSmResourceGroup_flags. */
};

/**
 * Sharing scope for workqueues
 */
enum __device_builtin__ cudaDevWorkqueueConfigScope {
    cudaDevWorkqueueConfigScopeDeviceCtx = 0, /**< Use all shared workqueue resources on the device. Default driver behaviour. */
    cudaDevWorkqueueConfigScopeGreenCtxBalanced = 1, /**< When possible, use non-overlapping workqueue resources with other balanced green contexts. */
};

/**
 * Data for workqueue configuration related resources
 */
struct __device_builtin__ cudaDevWorkqueueConfigResource {
    int device; /**< The device on which the workqueue resources are available */
    unsigned int wqConcurrencyLimit; /**< The expected maximum number of concurrent stream-ordered workloads */
    enum cudaDevWorkqueueConfigScope sharingScope; /**< The sharing scope for the workqueue resources */
};

/**
 * Handle to a pre-existing workqueue related resource
 */
struct __device_builtin__ cudaDevWorkqueueResource {
    unsigned char reserved[RESOURCE_ABI_BYTES]; /**< Reserved for future use */
};

/**
 * Input data for splitting SMs
 */
typedef __device_builtin__ struct cudaDevSmResourceGroupParams_st {
    unsigned int smCount;                       /**< The amount of SMs available in this resource. */
    unsigned int coscheduledSmCount;            /**< The amount of co-scheduled SMs grouped together for locality purposes. */
    unsigned int preferredCoscheduledSmCount;   /**< When possible, combine co-scheduled groups together into larger groups of this size. */
    unsigned int flags;                         /**< Combination of \p cudaDevSmResourceGroup_flags values to indicate this this group is created. */
    unsigned int reserved[12];                  /**< Reserved for future use - ensure this is is zero initialized. */
} cudaDevSmResourceGroupParams;

/**
 * A tagged union describing different resources identified by the type field. This structure should not be directly modified outside of the API that created it.
 * \code
 * struct {
 *     enum cudaDevResourceType type;
 *     union {
 *         struct cudaDevSmResource sm;
 *         struct cudaDevWorkqueueConfigResource wqConfig;
 *         struct cudaDevWorkqueueResource wq;
 *     };
 * };
 * \endcode
 * - If \p type is \p cudaDevResourceTypeInvalid, this resoure is not valid and cannot be further accessed.
 * - If \p type is \p cudaDevResourceTypeSm, the ::cudaDevSmResource structure \p sm is filled in. For example,
 * \p sm.smCount will reflect the amount of streaming multiprocessors available in this resource.
 * - If \p type is \p cudaDevResourceTypeWorkqueueConfig, the ::cudaDevWorkqueueConfigResource structure \p wqConfig is filled in.
 * - If \p type is \p cudaDevResourceTypeWorkqueue, the ::cudaDevWorkqueueResource structure \p wq is filled in.
 */
typedef __device_builtin__ struct cudaDevResource_st {
    enum cudaDevResourceType type; /**< Type of resource, dictates which union field was last set */
    unsigned char _internal_padding[92];
    union {
        struct cudaDevSmResource sm; /**< Resource corresponding to cudaDevResourceTypeSm \p type. */
        struct cudaDevWorkqueueConfigResource wqConfig; /**< Resource corresponding to cudaDevResourceTypeWorkqueueConfig \p type. */
        struct cudaDevWorkqueueResource wq; /**< Resource corresponding to cudaDevResourceTypeWorkqueue \p type. */
        unsigned char _oversize[RESOURCE_ABI_BYTES];
    };
    struct cudaDevResource_st* nextResource;
} cudaDevResource;

#undef RESOURCE_ABI_BYTES

/*******************************************************************************
*                                                                              *
*  SHORTHAND TYPE DEFINITION USED BY RUNTIME API                               *
*                                                                              *
*******************************************************************************/

/**
 * CUDA Error types
 */
typedef __device_builtin__ enum cudaError cudaError_t;

/**
 * CUDA stream
 */
typedef __device_builtin__ struct CUstream_st *cudaStream_t;

/**
 * CUDA event types
 */
typedef __device_builtin__ struct CUevent_st *cudaEvent_t;

/**
 * CUDA graphics resource types
 */
typedef __device_builtin__ struct cudaGraphicsResource *cudaGraphicsResource_t;

/**
 * CUDA external memory
 */
typedef __device_builtin__ struct CUexternalMemory_st *cudaExternalMemory_t;

/**
 * CUDA external semaphore
 */
typedef __device_builtin__ struct CUexternalSemaphore_st *cudaExternalSemaphore_t;

/**
 * CUDA graph
 */
typedef __device_builtin__ struct CUgraph_st *cudaGraph_t;

/**
 * CUDA graph node.
 */
typedef __device_builtin__ struct CUgraphNode_st *cudaGraphNode_t;

/**
 * CUDA user object for graphs
 */
typedef __device_builtin__ struct CUuserObject_st *cudaUserObject_t;

/**
 * CUDA handle for conditional graph nodes
 */
typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

/**
 * CUDA function
 */
typedef __device_builtin__ struct CUfunc_st *cudaFunction_t;

/**
 * CUDA kernel
 */
typedef __device_builtin__ struct CUkern_st *cudaKernel_t;

/**
 * Online compiler and linker options
 */
enum __device_builtin__ cudaJitOption
{
    /**
     * Max number of registers that a thread may use.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    cudaJitMaxRegisters = 0,

    /**
     * IN: Specifies minimum number of threads per block to target compilation
     * for\n
     * OUT: Returns the number of threads the compiler actually targeted.
     * This restricts the resource utilization of the compiler (e.g. max
     * registers) such that a block with the given number of threads should be
     * able to launch based on register limitations. Note, this option does not
     * currently take into account any other resource limitations, such as
     * shared memory utilization.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    cudaJitThreadsPerBlock = 1,

    /**
     * Overwrites the option value with the total wall clock time, in
     * milliseconds, spent in the compiler and linker\n
     * Option type: float\n
     * Applies to: compiler and linker
     */
    cudaJitWallTime = 2,

    /**
     * Pointer to a buffer in which to print any log messages
     * that are informational in nature (the buffer size is specified via
     * option ::cudaJitInfoLogBufferSizeBytes)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     */
    cudaJitInfoLogBuffer = 3,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int\n
     * Applies to: compiler and linker
     */
    cudaJitInfoLogBufferSizeBytes = 4,

    /**
     * Pointer to a buffer in which to print any log messages that
     * reflect errors (the buffer size is specified via option
     * ::cudaJitErrorLogBufferSizeBytes)\n
     * Option type: char *\n
     * Applies to: compiler and linker
     */
    cudaJitErrorLogBuffer = 5,

    /**
     * IN: Log buffer size in bytes.  Log messages will be capped at this size
     * (including null terminator)\n
     * OUT: Amount of log buffer filled with messages\n
     * Option type: unsigned int\n
     * Applies to: compiler and linker
     */
    cudaJitErrorLogBufferSizeBytes = 6,

    /**
     * Level of optimizations to apply to generated code (0 - 4), with 4
     * being the default and highest level of optimizations.\n
     * Option type: unsigned int\n
     * Applies to: compiler only
     */
    cudaJitOptimizationLevel = 7,

    /**
     * Specifies choice of fallback strategy if matching cubin is not found.
     * Choice is based on supplied ::cudaJit_Fallback.
     * Option type: unsigned int for enumerated type ::cudaJit_Fallback\n
     * Applies to: compiler only
     */
    cudaJitFallbackStrategy = 10,

    /**
     * Specifies whether to create debug information in output (-g)
     * (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler and linker
     */
    cudaJitGenerateDebugInfo = 11,

    /**
     * Generate verbose log messages (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler and linker
     */
    cudaJitLogVerbose = 12,

    /**
     * Generate line number information (-lineinfo) (0: false, default)\n
     * Option type: int\n
     * Applies to: compiler only
     */
    cudaJitGenerateLineInfo = 13,

    /**
     * Specifies whether to enable caching explicitly (-dlcm) \n
     * Choice is based on supplied ::cudaJit_CacheMode.\n
     * Option type: unsigned int for enumerated type ::cudaJit_CacheMode\n
     * Applies to: compiler only
     */
    cudaJitCacheMode = 14,

    /**
     * Generate position independent code (0: false)\n
     * Option type: int\n
     * Applies to: compiler only
     */
    cudaJitPositionIndependentCode = 30,

    /**
     * This option hints to the JIT compiler the minimum number of CTAs from the
     * kernel’s grid to be mapped to a SM. This option is ignored when used together
     * with ::cudaJitMaxRegisters or ::cudaJitThreadsPerBlock.
     * Optimizations based on this option need ::cudaJitMaxThreadsPerBlock to
     * be specified as well. For kernels already using PTX directive .minnctapersm,
     * this option will be ignored by default. Use ::cudaJitOverrideDirectiveValues
     * to let this option take precedence over the PTX directive.
     * Option type: unsigned int\n
     * Applies to: compiler only
    */
    cudaJitMinCtaPerSm = 31,

     /**
     * Maximum number threads in a thread block, computed as the product of
     * the maximum extent specifed for each dimension of the block. This limit
     * is guaranteed not to be exeeded in any invocation of the kernel. Exceeding
     * the the maximum number of threads results in runtime error or kernel launch
     * failure. For kernels already using PTX directive .maxntid, this option will
     * be ignored by default. Use ::cudaJitOverrideDirectiveValues to let this
     * option take precedence over the PTX directive.
     * Option type: int\n
     * Applies to: compiler only
    */
    cudaJitMaxThreadsPerBlock = 32,

    /**
     * This option lets the values specified using ::cudaJitMaxRegisters,
     * ::cudaJitThreadsPerBlock, ::cudaJitMaxThreadsPerBlock and
     * ::cudaJitMinCtaPerSm take precedence over any PTX directives.
     * (0: Disable, default; 1: Enable)
     * Option type: int\n
     * Applies to: compiler only
    */
    cudaJitOverrideDirectiveValues = 33,
};


/**
 * Library options to be specified with ::cudaLibraryLoadData() or ::cudaLibraryLoadFromFile()
 */
enum __device_builtin__ cudaLibraryOption
{
    cudaLibraryHostUniversalFunctionAndDataTable = 0,

    /**
     * Specifes that the argument \p code passed to ::cudaLibraryLoadData() will be preserved.
     * Specifying this option will let the driver know that \p code can be accessed at any point
     * until ::cudaLibraryUnload(). The default behavior is for the driver to allocate and
     * maintain its own copy of \p code. Note that this is only a memory usage optimization
     * hint and the driver can choose to ignore it if required.
     * Specifying this option with ::cudaLibraryLoadFromFile() is invalid and
     * will return ::cudaErrorInvalidValue.
     */
    cudaLibraryBinaryIsPreserved = 1,
};

struct __device_builtin__ cudalibraryHostUniversalFunctionAndDataTable
{
    void *functionTable;
    size_t functionWindowSize;
    void *dataTable;
    size_t dataWindowSize;
};

/**
 * Caching modes for dlcm
 */
enum __device_builtin__ cudaJit_CacheMode
{
    cudaJitCacheOptionNone = 0,   /**< Compile with no -dlcm flag specified */
    cudaJitCacheOptionCG,         /**< Compile with L1 cache disabled */
    cudaJitCacheOptionCA          /**< Compile with L1 cache enabled */
};

/**
 * Cubin matching fallback strategies
 */
enum __device_builtin__ cudaJit_Fallback
{
    cudaPreferPtx = 0,  /**< Prefer to compile ptx if exact binary match not found */

    cudaPreferBinary    /**< Prefer to fall back to compatible binary code if exact match not found */
};

/**
 * CUDA library
 */
typedef __device_builtin__ struct CUlib_st *cudaLibrary_t;

/**
 * CUDA memory pool
 */
typedef __device_builtin__ struct CUmemPoolHandle_st *cudaMemPool_t;

/**
 * CUDA cooperative group scope
 */
enum __device_builtin__ cudaCGScope {
    cudaCGScopeInvalid   = 0, /**< Invalid cooperative group scope */
    cudaCGScopeGrid      = 1, /**< Scope represented by a grid_group */
    cudaCGScopeReserved  = 2  /**< Reserved */
};

/**
 * CUDA GPU kernel node parameters
 */
struct __device_builtin__ cudaKernelNodeParams {
    void* func;                     /**< Kernel to launch */
    dim3 gridDim;                   /**< Grid dimensions */
    dim3 blockDim;                  /**< Block dimensions */
    unsigned int sharedMemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    void **kernelParams;            /**< Array of pointers to individual kernel arguments*/
    void **extra;                   /**< Pointer to kernel arguments in the "extra" format */
};

/**
 * CUDA GPU kernel node parameters
 */
struct __device_builtin__ cudaKernelNodeParamsV2 {
    void* func;                     /**< Kernel to launch */
    #if !defined(__cplusplus) || __cplusplus >= 201103L
        dim3 gridDim;                   /**< Grid dimensions */
        dim3 blockDim;                  /**< Block dimensions */
    #else
        /* Union members cannot have nontrivial constructors until C++11. */
        uint3 gridDim;                  /**< Grid dimensions */
        uint3 blockDim;                 /**< Block dimensions */
    #endif
    unsigned int sharedMemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    void **kernelParams;            /**< Array of pointers to individual kernel arguments*/
    void **extra;                   /**< Pointer to kernel arguments in the "extra" format */
    cudaExecutionContext_t ctx;     /**< Context in which to run the kernel. If NULL will try to use the current context. */
};

/**
 * External semaphore signal node parameters
 */
struct __device_builtin__ cudaExternalSemaphoreSignalNodeParams {
    cudaExternalSemaphore_t* extSemArray;                        /**< Array of external semaphore handles. */
    const struct cudaExternalSemaphoreSignalParams* paramsArray; /**< Array of external semaphore signal parameters. */
    unsigned int numExtSems;                                     /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore signal node parameters
 */
struct __device_builtin__ cudaExternalSemaphoreSignalNodeParamsV2 {
    cudaExternalSemaphore_t* extSemArray;                        /**< Array of external semaphore handles. */
    const struct cudaExternalSemaphoreSignalParams* paramsArray; /**< Array of external semaphore signal parameters. */
    unsigned int numExtSems;                                     /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore wait node parameters
 */
struct __device_builtin__ cudaExternalSemaphoreWaitNodeParams {
    cudaExternalSemaphore_t* extSemArray;                      /**< Array of external semaphore handles. */
    const struct cudaExternalSemaphoreWaitParams* paramsArray; /**< Array of external semaphore wait parameters. */
    unsigned int numExtSems;                                   /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

/**
 * External semaphore wait node parameters
 */
struct __device_builtin__ cudaExternalSemaphoreWaitNodeParamsV2 {
    cudaExternalSemaphore_t* extSemArray;                      /**< Array of external semaphore handles. */
    const struct cudaExternalSemaphoreWaitParams* paramsArray; /**< Array of external semaphore wait parameters. */
    unsigned int numExtSems;                                   /**< Number of handles and parameters supplied in extSemArray and paramsArray. */
};

enum __device_builtin__ cudaGraphConditionalHandleFlags {
    cudaGraphCondAssignDefault = 1 /**< Apply default handle value when graph is launched. */
};

/**
 * CUDA conditional node types
 */
enum __device_builtin__ cudaGraphConditionalNodeType {
    cudaGraphCondTypeIf  = 0,    /**< Conditional 'if/else' Node. Body[0] executed if condition is non-zero.  If \p size == 2, an optional ELSE graph is created and this is executed if the condition is zero. */
    cudaGraphCondTypeWhile = 1,  /**< Conditional 'while' Node. Body executed repeatedly while condition value is non-zero. */
    cudaGraphCondTypeSwitch = 2, /**< Conditional 'switch' Node. Body[n] is executed once, where 'n' is the value of the condition. If the condition does not match a body index, no body is launched. */
};

/**
 * CUDA conditional node parameters
 */
struct __device_builtin__ cudaConditionalNodeParams {
    cudaGraphConditionalHandle handle;       /**< Conditional node handle.
                                                  Handles must be created in advance of creating the node
                                                  using ::cudaGraphConditionalHandleCreate. */
    enum cudaGraphConditionalNodeType type;  /**< Type of conditional node. */
    unsigned int size;                       /**< Size of graph output array.  Allowed values are 1 for cudaGraphCondTypeWhile, 1 or 2
                                                  for cudaGraphCondTypeIf, or any value greater than zero for cudaGraphCondTypeSwitch. */
    cudaGraph_t *phGraph_out;                /**< CUDA-owned array populated with conditional node child graphs during creation of the node.
                                                  Valid for the lifetime of the conditional node.
                                                  The contents of the graph(s) are subject to the following constraints:
                                                  
                                                  - Allowed node types are kernel nodes, empty nodes, child graphs, memsets,
                                                    memcopies, and conditionals. This applies recursively to child graphs and conditional bodies.
                                                  - All kernels, including kernels in nested conditionals or child graphs at any level,
                                                    must belong to the same CUDA context.
                                                  
                                                  These graphs may be populated using graph node creation APIs or ::cudaStreamBeginCaptureToGraph.
                                                  cudaGraphCondTypeIf:
                                                  phGraph_out[0] is executed when the condition is non-zero.  If \p size == 2, phGraph_out[1] will
                                                  be executed when the condition is zero.
                                                  cudaGraphCondTypeWhile:
                                                  phGraph_out[0] is executed as long as the condition is non-zero.
                                                  cudaGraphCondTypeSwitch:
                                                  phGraph_out[n] is executed when the condition is equal to n.  If the condition >= \p size,
                                                  no body graph is executed.
                                         */
    cudaExecutionContext_t ctx;              /**< CUDA Execution Context */
};

/**
* CUDA Graph node types
*/
enum __device_builtin__ cudaGraphNodeType {
    cudaGraphNodeTypeKernel      = 0x00, /**< GPU kernel node */
    cudaGraphNodeTypeMemcpy      = 0x01, /**< Memcpy node */
    cudaGraphNodeTypeMemset      = 0x02, /**< Memset node */
    cudaGraphNodeTypeHost        = 0x03, /**< Host (executable) node */
    cudaGraphNodeTypeGraph       = 0x04, /**< Node which executes an embedded graph */
    cudaGraphNodeTypeEmpty       = 0x05, /**< Empty (no-op) node */
    cudaGraphNodeTypeWaitEvent   = 0x06, /**< External event wait node */
    cudaGraphNodeTypeEventRecord = 0x07, /**< External event record node */
    cudaGraphNodeTypeExtSemaphoreSignal = 0x08, /**< External semaphore signal node */
    cudaGraphNodeTypeExtSemaphoreWait = 0x09, /**< External semaphore wait node */
    cudaGraphNodeTypeMemAlloc    = 0x0a, /**< Memory allocation node */
    cudaGraphNodeTypeMemFree     = 0x0b, /**< Memory free node */
    cudaGraphNodeTypeConditional = 0x0d, /**< Conditional node
                                              
                                              May be used to implement a conditional execution path or loop
                                              inside of a graph. The graph(s) contained within the body of the conditional node
                                              can be selectively executed or iterated upon based on the value of a conditional
                                              variable.
                                              
                                              Handles must be created in advance of creating the node
                                              using ::cudaGraphConditionalHandleCreate.
                                              
                                              The following restrictions apply to graphs which contain conditional nodes:
                                                The graph cannot be used in a child node.
                                                Only one instantiation of the graph may exist at any point in time.
                                                The graph cannot be cloned.
                                              
                                              To set the control value, supply a default value when creating the handle and/or
                                              call ::cudaGraphSetConditional from device code.*/
    cudaGraphNodeTypeCount
};

/**
 * Child graph node ownership
 */
enum __device_builtin__ cudaGraphChildGraphNodeOwnership {
    cudaGraphChildGraphOwnershipClone = 0,      /**< Default behavior for a child graph node. Child graph is cloned
                                                     into the parent and memory allocation/free nodes can't be present
                                                     in the child graph. */
    cudaGraphChildGraphOwnershipMove  = 1,      /**< The child graph is moved to the parent. The handle to the child graph
                                                     is owned by the parent and will be destroyed when the parent is
                                                     destroyed.

                                                     The following restrictions apply to child graphs after they have been moved:
                                                      Cannot be independently instantiated or destroyed;
                                                      Cannot be added as a child graph of a separate parent graph;
                                                      Cannot be used as an argument to cudaGraphExecUpdate;
                                                      Cannot have additional memory allocation or free nodes added. */
};

/**
 * Child graph node parameters
 */
struct __device_builtin__ cudaChildGraphNodeParams {
    cudaGraph_t graph; /**< The child graph to clone into the node for node creation, or
                        *   a handle to the graph owned by the node for node query.
                        *   The graph must not contain conditional nodes. Graphs
                        *   containing memory allocation or memory free nodes must
                        *   set the ownership to be moved to the parent.
                        */
    enum cudaGraphChildGraphNodeOwnership ownership; /**< The ownership relationship of the child graph node. */
};

/**
 * Event record node parameters
 */
struct __device_builtin__ cudaEventRecordNodeParams {
    cudaEvent_t event; /**< The event to record when the node executes */
};

/**
 * Event wait node parameters
 */
struct __device_builtin__ cudaEventWaitNodeParams {
    cudaEvent_t event; /**< The event to wait on from the node */
};

/**
 * Graph node parameters.  See ::cudaGraphAddNode.
 */
struct __device_builtin__ cudaGraphNodeParams {
    enum cudaGraphNodeType type; /**< Type of the node */
    int reserved0[3];            /**< Reserved.  Must be zero. */

    union {
        long long                                      reserved1[29]; /**< Padding. Unused bytes must be zero. */
        struct cudaKernelNodeParamsV2                  kernel;        /**< Kernel node parameters. */
        struct cudaMemcpyNodeParams                    memcpy;        /**< Memcpy node parameters. */
        struct cudaMemsetParamsV2                      memset;        /**< Memset node parameters. */
        struct cudaHostNodeParamsV2                    host;          /**< Host node parameters. */
        struct cudaChildGraphNodeParams                graph;         /**< Child graph node parameters. */
        struct cudaEventWaitNodeParams                 eventWait;     /**< Event wait node parameters. */
        struct cudaEventRecordNodeParams               eventRecord;   /**< Event record node parameters. */
        struct cudaExternalSemaphoreSignalNodeParamsV2 extSemSignal;  /**< External semaphore signal node parameters. */
        struct cudaExternalSemaphoreWaitNodeParamsV2   extSemWait;    /**< External semaphore wait node parameters. */
        struct cudaMemAllocNodeParamsV2                alloc;         /**< Memory allocation node parameters. */
        struct cudaMemFreeNodeParams                   free;          /**< Memory free node parameters. */
        struct cudaConditionalNodeParams               conditional;   /**< Conditional node parameters. */
    };

    long long reserved2; /**< Reserved bytes. Must be zero. */
};

/**
 * Type annotations that can be applied to graph edges as part of ::cudaGraphEdgeData.
 */
typedef __device_builtin__ enum cudaGraphDependencyType_enum {
    cudaGraphDependencyTypeDefault = 0, /**< This is an ordinary dependency. */
    cudaGraphDependencyTypeProgrammatic = 1  /**< This dependency type allows the downstream node to
                                                  use \c cudaGridDependencySynchronize(). It may only be used
                                                  between kernel nodes, and must be used with either the
                                                  ::cudaGraphKernelNodePortProgrammatic or
                                                  ::cudaGraphKernelNodePortLaunchCompletion outgoing port. */
} cudaGraphDependencyType;

/**
 * Optional annotation for edges in a CUDA graph. Note, all edges implicitly have annotations and
 * default to a zero-initialized value if not specified. A zero-initialized struct indicates a
 * standard full serialization of two nodes with memory visibility.
 */
typedef __device_builtin__ struct cudaGraphEdgeData_st {
    unsigned char from_port; /**< This indicates when the dependency is triggered from the upstream
                                  node on the edge. The meaning is specfic to the node type. A value
                                  of 0 in all cases means full completion of the upstream node, with
                                  memory visibility to the downstream node or portion thereof
                                  (indicated by \c to_port).
                                  <br>
                                  Only kernel nodes define non-zero ports. A kernel node
                                  can use the following output port types:
                                  ::cudaGraphKernelNodePortDefault, ::cudaGraphKernelNodePortProgrammatic,
                                  or ::cudaGraphKernelNodePortLaunchCompletion. */
    unsigned char to_port; /**< This indicates what portion of the downstream node is dependent on
                                the upstream node or portion thereof (indicated by \c from_port). The
                                meaning is specific to the node type. A value of 0 in all cases means
                                the entirety of the downstream node is dependent on the upstream work.
                                <br>
                                Currently no node types define non-zero ports. Accordingly, this field
                                must be set to zero. */
    unsigned char type; /**< This should be populated with a value from ::cudaGraphDependencyType. (It
                             is typed as char due to compiler-specific layout of bitfields.) See
                             ::cudaGraphDependencyType. */
    unsigned char reserved[5]; /**< These bytes are unused and must be zeroed. This ensures
                                    compatibility if additional fields are added in the future. */
} cudaGraphEdgeData;

/**
 * This port activates when the kernel has finished executing.
 */
#define cudaGraphKernelNodePortDefault 0
/**
 * This port activates when all blocks of the kernel have performed cudaTriggerProgrammaticLaunchCompletion()
 * or have terminated. It must be used with edge type ::cudaGraphDependencyTypeProgrammatic. See also
 * ::cudaLaunchAttributeProgrammaticEvent.
 */
#define cudaGraphKernelNodePortProgrammatic 1
/**
 * This port activates when all blocks of the kernel have begun execution. See also
 * ::cudaLaunchAttributeLaunchCompletionEvent.
 */
#define cudaGraphKernelNodePortLaunchCompletion 2

/**
 * CUDA executable (launchable) graph
 */
typedef struct CUgraphExec_st* cudaGraphExec_t;

/**
* CUDA Graph Update error types
*/
enum __device_builtin__ cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess                = 0x0, /**< The update succeeded */
    cudaGraphExecUpdateError                  = 0x1, /**< The update failed for an unexpected reason which is described in the return value of the function */
    cudaGraphExecUpdateErrorTopologyChanged   = 0x2, /**< The update failed because the topology changed */
    cudaGraphExecUpdateErrorNodeTypeChanged   = 0x3, /**< The update failed because a node type changed */
    cudaGraphExecUpdateErrorFunctionChanged   = 0x4, /**< The update failed because the function of a kernel node changed (CUDA driver < 11.2) */
    cudaGraphExecUpdateErrorParametersChanged = 0x5, /**< The update failed because the parameters changed in a way that is not supported */
    cudaGraphExecUpdateErrorNotSupported      = 0x6, /**< The update failed because something about the node is not supported */
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 0x7, /**< The update failed because the function of a kernel node changed in an unsupported way */
    cudaGraphExecUpdateErrorAttributesChanged = 0x8 /**< The update failed because the node attributes changed in a way that is not supported */
};

/**
 * Graph instantiation results
*/
typedef __device_builtin__ enum cudaGraphInstantiateResult {
    cudaGraphInstantiateSuccess = 0,                       /**< Instantiation succeeded */
    cudaGraphInstantiateError = 1,                         /**< Instantiation failed for an unexpected reason which is described in the return value of the function */
    cudaGraphInstantiateInvalidStructure = 2,              /**< Instantiation failed due to invalid structure, such as cycles */
    cudaGraphInstantiateNodeOperationNotSupported = 3,     /**< Instantiation for device launch failed because the graph contained an unsupported operation */
    cudaGraphInstantiateMultipleDevicesNotSupported = 4,   /**< Instantiation for device launch failed due to the nodes belonging to different contexts */
    cudaGraphInstantiateConditionalHandleUnused = 5        /**< One or more conditional handles are not associated with conditional nodes */
} cudaGraphInstantiateResult;

/**
 * Graph instantiation parameters
 */
typedef __device_builtin__ struct cudaGraphInstantiateParams_st
{
    unsigned long long flags;              /**< Instantiation flags */
    cudaStream_t uploadStream;             /**< Upload stream */
    cudaGraphNode_t errNode_out;           /**< The node which caused instantiation to fail, if any */
    cudaGraphInstantiateResult result_out; /**< Whether instantiation was successful.  If it failed, the reason why */
} cudaGraphInstantiateParams;

/**
 * Result information returned by cudaGraphExecUpdate
 */
typedef __device_builtin__ struct cudaGraphExecUpdateResultInfo_st {
    /**
     * Gives more specific detail when a cuda graph update fails. 
     */
    enum cudaGraphExecUpdateResult result;

    /**
     * The "to node" of the error edge when the topologies do not match.
     * The error node when the error is associated with a specific node.
     * NULL when the error is generic.
     */
    cudaGraphNode_t errorNode;

    /**
     * The from node of error edge when the topologies do not match. Otherwise NULL.
     */
    cudaGraphNode_t errorFromNode;
} cudaGraphExecUpdateResultInfo;

/**
 * CUDA device node handle for device-side node update
 */
typedef struct CUgraphDeviceUpdatableNode_st* cudaGraphDeviceNode_t;

/**
 * Specifies the field to update when performing multiple node updates from the device
 */
enum __device_builtin__ cudaGraphKernelNodeField
{
    cudaGraphKernelNodeFieldInvalid = 0, /**< Invalid field */
    cudaGraphKernelNodeFieldGridDim,     /**< Grid dimension update */
    cudaGraphKernelNodeFieldParam,       /**< Kernel parameter update */
    cudaGraphKernelNodeFieldEnabled      /**< Node enable/disable */
};

/**
 * Struct to specify a single node update to pass as part of a larger array to ::cudaGraphKernelNodeUpdatesApply
 */
struct __device_builtin__ cudaGraphKernelNodeUpdate {
    cudaGraphDeviceNode_t node;     /**< Node to update */
    enum cudaGraphKernelNodeField field; /**< Which type of update to apply. Determines how updateData is interpreted */
    union {
#if !defined(__cplusplus) || __cplusplus >= 201103L
        dim3 gridDim;               /**< Grid dimensions */
#else
        /* Union members cannot have nontrivial constructors until C++11. */
        uint3 gridDim;              /**< Grid dimensions */
#endif
        struct {
            const void *pValue;     /**< Kernel parameter data to write in */
            size_t offset;          /**< Offset into the parameter buffer at which to apply the update */
            size_t size;            /**< Number of bytes to update */
        } param;                    /**< Kernel parameter data */
        unsigned int isEnabled;     /**< Node enable/disable data. Nonzero if the node should be enabled, 0 if it should be disabled */
    } updateData;                   /**< Update data to apply. Which field is used depends on field's value */
};

/**
 * Flags to specify search options to be used with ::cudaGetDriverEntryPoint
 * For more details see ::cuGetProcAddress
 */ 
enum __device_builtin__ cudaGetDriverEntryPointFlags {
    cudaEnableDefault                = 0x0, /**< Default search mode for driver symbols. */
    cudaEnableLegacyStream           = 0x1, /**< Search for legacy versions of driver symbols. */
    cudaEnablePerThreadDefaultStream = 0x2  /**< Search for per-thread versions of driver symbols. */
};

/**
 * Enum for status from obtaining driver entry points, used with ::cudaApiGetDriverEntryPoint
 */
enum __device_builtin__ cudaDriverEntryPointQueryResult {
    cudaDriverEntryPointSuccess             = 0,  /**< Search for symbol found a match */
    cudaDriverEntryPointSymbolNotFound      = 1,  /**< Search for symbol was not found */
    cudaDriverEntryPointVersionNotSufficent = 2   /**< Search for symbol was found but version wasn't great enough */
};

/**
 * CUDA Graph debug write options
 */
enum __device_builtin__ cudaGraphDebugDotFlags {
    cudaGraphDebugDotFlagsVerbose                  = 1<<0,  /**< Output all debug data as if every debug flag is enabled */
    cudaGraphDebugDotFlagsKernelNodeParams         = 1<<2,  /**< Adds cudaKernelNodeParams to output */
    cudaGraphDebugDotFlagsMemcpyNodeParams         = 1<<3,  /**< Adds cudaMemcpy3DParms to output */
    cudaGraphDebugDotFlagsMemsetNodeParams         = 1<<4,  /**< Adds cudaMemsetParams to output */
    cudaGraphDebugDotFlagsHostNodeParams           = 1<<5,  /**< Adds cudaHostNodeParams to output */
    cudaGraphDebugDotFlagsEventNodeParams          = 1<<6,  /**< Adds cudaEvent_t handle from record and wait nodes to output */
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 1<<7,  /**< Adds cudaExternalSemaphoreSignalNodeParams values to output */
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams   = 1<<8,  /**< Adds cudaExternalSemaphoreWaitNodeParams to output */
    cudaGraphDebugDotFlagsKernelNodeAttributes     = 1<<9,  /**< Adds cudaKernelNodeAttrID values to output */
    cudaGraphDebugDotFlagsHandles                  = 1<<10, /**< Adds node handles and every kernel function handle to output */
    cudaGraphDebugDotFlagsConditionalNodeParams    = 1<<15, /**< Adds cudaConditionalNodeParams to output */
};

/**
 * Flags for instantiating a graph
 */
enum __device_builtin__ cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1 /**< Automatically free memory allocated in a graph before relaunching. */
  , cudaGraphInstantiateFlagUpload           = 2 /**< Automatically upload the graph after instantiation. Only supported by                                                                                                                                                                                                                                                                                                     
                                                      ::cudaGraphInstantiateWithParams.  The upload will be performed using the                                                                                                                                                                                                                                                                                                   
                                                      stream provided in \p instantiateParams. */                                                                                                                                                                                                                                                                                                                               
  , cudaGraphInstantiateFlagDeviceLaunch     = 4 /**< Instantiate the graph to be launchable from the device. This flag can only                                                                                                                                                                                                                                                                                                
                                                      be used on platforms which support unified addressing. This flag cannot be                                                                                                                                                                                                                                                                                                
                                                      used in conjunction with cudaGraphInstantiateFlagAutoFreeOnLaunch. */                                                                                                                                                                                                                                                                                              
  , cudaGraphInstantiateFlagUseNodePriority  = 8 /**< Run the graph using the per-node priority attributes rather than the
                                                      priority of the stream it is launched into. */
};

/**
 * Memory Synchronization Domain
 *
 * A kernel can be launched in a specified memory synchronization domain that affects all memory operations issued by
 * that kernel. A memory barrier issued in one domain will only order memory operations in that domain, thus eliminating
 * latency increase from memory barriers ordering unrelated traffic.
 *
 * By default, kernels are launched in domain 0. Kernel launched with ::cudaLaunchMemSyncDomainRemote will have a
 * different domain ID. User may also alter the domain ID with ::cudaLaunchMemSyncDomainMap for a specific stream /
 * graph node / kernel launch. See ::cudaLaunchAttributeMemSyncDomain, ::cudaStreamSetAttribute, ::cudaLaunchKernelEx,
 * ::cudaGraphKernelNodeSetAttribute.
 *
 * Memory operations done in kernels launched in different domains are considered system-scope distanced. In other
 * words, a GPU scoped memory synchronization is not sufficient for memory order to be observed by kernels in another
 * memory synchronization domain even if they are on the same GPU.
 */
typedef __device_builtin__ enum cudaLaunchMemSyncDomain {
    cudaLaunchMemSyncDomainDefault = 0,    /**< Launch kernels in the default domain */
    cudaLaunchMemSyncDomainRemote  = 1     /**< Launch kernels in the remote domain */
} cudaLaunchMemSyncDomain;

/**
 * Memory Synchronization Domain map
 *
 * See ::cudaLaunchMemSyncDomain.
 *
 * By default, kernels are launched in domain 0. Kernel launched with ::cudaLaunchMemSyncDomainRemote will have a
 * different domain ID. User may also alter the domain ID with ::cudaLaunchMemSyncDomainMap for a specific stream /
 * graph node / kernel launch. See ::cudaLaunchAttributeMemSyncDomainMap.
 *
 * Domain ID range is available through ::cudaDevAttrMemSyncDomainCount.
 */
typedef __device_builtin__ struct cudaLaunchMemSyncDomainMap_st {
    unsigned char default_;                /**< The default domain ID to use for designated kernels */
    unsigned char remote;                  /**< The remote domain ID to use for designated kernels */
} cudaLaunchMemSyncDomainMap;

/**
 * Launch attributes enum; used as id field of ::cudaLaunchAttribute
 */
typedef __device_builtin__ enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore                = 0 /**< Ignored entry, for convenient composition */
  , cudaLaunchAttributeAccessPolicyWindow    = 1 /**< Valid for streams, graph nodes, launches. See
                                                    ::cudaLaunchAttributeValue::accessPolicyWindow. */
  , cudaLaunchAttributeCooperative           = 2 /**< Valid for graph nodes, launches. See
                                                    ::cudaLaunchAttributeValue::cooperative. */
  , cudaLaunchAttributeSynchronizationPolicy = 3 /**< Valid for streams. See ::cudaLaunchAttributeValue::syncPolicy. */
  , cudaLaunchAttributeClusterDimension                  = 4 /**< Valid for graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::clusterDim. */
  , cudaLaunchAttributeClusterSchedulingPolicyPreference = 5 /**< Valid for graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::clusterSchedulingPolicyPreference. */
  , cudaLaunchAttributeProgrammaticStreamSerialization   = 6 /**< Valid for launches. Setting
                                                                  ::cudaLaunchAttributeValue::programmaticStreamSerializationAllowed
                                                                  to non-0 signals that the kernel will use programmatic
                                                                  means to resolve its stream dependency, so that the
                                                                  CUDA runtime should opportunistically allow the grid's
                                                                  execution to overlap with the previous kernel in the
                                                                  stream, if that kernel requests the overlap. The
                                                                  dependent launches can choose to wait on the
                                                                  dependency using the programmatic sync
                                                                  (cudaGridDependencySynchronize() or equivalent PTX
                                                                  instructions). */
  , cudaLaunchAttributeProgrammaticEvent                 = 7 /**< Valid for launches. Set
                                                                  ::cudaLaunchAttributeValue::programmaticEvent to
                                                                  record the event. Event recorded through this launch
                                                                  attribute is guaranteed to only trigger after all
                                                                  block in the associated kernel trigger the event.  A
                                                                  block can trigger the event programmatically in a
                                                                  future CUDA release. A trigger can also be inserted at
                                                                  the beginning of each block's execution if
                                                                  triggerAtBlockStart is set to non-0. The dependent
                                                                  launches can choose to wait on the dependency using
                                                                  the programmatic sync (cudaGridDependencySynchronize()
                                                                  or equivalent PTX instructions). Note that dependents
                                                                  (including the CPU thread calling
                                                                  cudaEventSynchronize()) are not guaranteed to observe
                                                                  the release precisely when it is released. For
                                                                  example, cudaEventSynchronize() may only observe the
                                                                  event trigger long after the associated kernel has
                                                                  completed. This recording type is primarily meant for
                                                                  establishing programmatic dependency between device
                                                                  tasks. Note also this type of dependency allows, but
                                                                  does not guarantee, concurrent execution of tasks.
                                                                  <br>
                                                                  The event supplied must not be an interprocess or
                                                                  interop event. The event must disable timing (i.e.
                                                                  must be created with the ::cudaEventDisableTiming flag
                                                                  set). */
  , cudaLaunchAttributePriority              = 8 /**< Valid for streams, graph nodes, launches. See
                                                    ::cudaLaunchAttributeValue::priority. */
  , cudaLaunchAttributeMemSyncDomainMap                  = 9 /**< Valid for streams, graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::memSyncDomainMap. */
  , cudaLaunchAttributeMemSyncDomain                    = 10 /**< Valid for streams, graph nodes, launches. See
                                                                ::cudaLaunchAttributeValue::memSyncDomain. */
  , cudaLaunchAttributePreferredClusterDimension = 11 /**< Valid for graph nodes and launches. Set
                                                           ::cudaLaunchAttributeValue::preferredClusterDim
                                                           to allow the kernel launch to specify a preferred substitute
                                                           cluster dimension. Blocks may be grouped according to either
                                                           the dimensions specified with this attribute (grouped into a
                                                           "preferred substitute cluster"), or the one specified with
                                                           ::cudaLaunchAttributeClusterDimension attribute (grouped
                                                           into a "regular cluster"). The cluster dimensions of a
                                                           "preferred substitute cluster" shall be an integer multiple
                                                           greater than zero of the regular cluster dimensions. The
                                                           device will attempt - on a best-effort basis - to group
                                                           thread blocks into preferred clusters over grouping them
                                                           into regular clusters. When it deems necessary (primarily
                                                           when the device temporarily runs out of physical resources
                                                           to launch the larger preferred clusters), the device may
                                                           switch to launch the regular clusters instead to attempt to
                                                           utilize as much of the physical device resources as possible.
                                                           <br>
                                                           Each type of cluster will have its enumeration / coordinate
                                                           setup as if the grid consists solely of its type of cluster.
                                                           For example, if the preferred substitute cluster dimensions
                                                           double the regular cluster dimensions, there might be
                                                           simultaneously a regular cluster indexed at (1,0,0), and a
                                                           preferred cluster indexed at (1,0,0). In this example, the
                                                           preferred substitute cluster (1,0,0) replaces regular
                                                           clusters (2,0,0) and (3,0,0) and groups their blocks.
                                                           <br>
                                                           This attribute will only take effect when a regular cluster
                                                           dimension has been specified. The preferred substitute cluster
                                                           dimension must be an integer multiple greater than zero of the
                                                           regular cluster dimension and must divide the grid. It must
                                                           also be no more than `maxBlocksPerCluster`, if it is set in
                                                           the kernel's `__launch_bounds__`. Otherwise it must be less
                                                           than the maximum value the driver can support. Otherwise,
                                                           setting this attribute to a value physically unable to fit on
                                                           any particular device is permitted. */
  , cudaLaunchAttributeLaunchCompletionEvent = 12 /**< Valid for launches. Set
                                                       ::cudaLaunchAttributeValue::launchCompletionEvent to record the
                                                       event.
                                                       <br>
                                                       Nominally, the event is triggered once all blocks of the kernel
                                                       have begun execution. Currently this is a best effort. If a kernel
                                                       B has a launch completion dependency on a kernel A, B may wait
                                                       until A is complete. Alternatively, blocks of B may begin before
                                                       all blocks of A have begun, for example if B can claim execution
                                                       resources unavailable to A (e.g. they run on different GPUs) or
                                                       if B is a higher priority than A.
                                                       Exercise caution if such an ordering inversion could lead
                                                       to deadlock.
                                                       <br>
                                                       A launch completion event is nominally similar to a programmatic
                                                       event with \c triggerAtBlockStart set except that it is not
                                                       visible to \c cudaGridDependencySynchronize() and can be used with
                                                       compute capability less than 9.0.
                                                       <br>
                                                       The event supplied must not be an interprocess or interop event.
                                                       The event must disable timing (i.e. must be created with the
                                                       ::cudaEventDisableTiming flag set). */
  , cudaLaunchAttributeDeviceUpdatableKernelNode = 13 /**< Valid for graph nodes, launches. This attribute is graphs-only,
                                                           and passing it to a launch in a non-capturing stream will result
                                                           in an error.
                                                           <br>
                                                           :cudaLaunchAttributeValue::deviceUpdatableKernelNode::deviceUpdatable can 
                                                           only be set to 0 or 1. Setting the field to 1 indicates that the
                                                           corresponding kernel node should be device-updatable. On success, a handle
                                                           will be returned via
                                                           ::cudaLaunchAttributeValue::deviceUpdatableKernelNode::devNode which can be
                                                           passed to the various device-side update functions to update the node's
                                                           kernel parameters from within another kernel. For more information on the
                                                           types of device updates that can be made, as well as the relevant limitations
                                                           thereof, see ::cudaGraphKernelNodeUpdatesApply.
                                                           <br>
                                                           Nodes which are device-updatable have additional restrictions compared to
                                                           regular kernel nodes. Firstly, device-updatable nodes cannot be removed
                                                           from their graph via ::cudaGraphDestroyNode. Additionally, once opted-in
                                                           to this functionality, a node cannot opt out, and any attempt to set the
                                                           deviceUpdatable attribute to 0 will result in an error. Device-updatable
                                                           kernel nodes also cannot have their attributes copied to/from another kernel
                                                           node via ::cudaGraphKernelNodeCopyAttributes. Graphs containing one or more
                                                           device-updatable nodes also do not allow multiple instantiation, and neither
                                                           the graph nor its instantiated version can be passed to ::cudaGraphExecUpdate.
                                                           <br>
                                                           If a graph contains device-updatable nodes and updates those nodes from the device
                                                           from within the graph, the graph must be uploaded with ::cuGraphUpload before it
                                                           is launched. For such a graph, if host-side executable graph updates are made to the
                                                           device-updatable nodes, the graph must be uploaded before it is launched again. */
  , cudaLaunchAttributePreferredSharedMemoryCarveout = 14 /**< Valid for launches. On devices where the L1 cache and shared memory use the
                                                               same hardware resources, setting ::cudaLaunchAttributeValue::sharedMemCarveout 
                                                               to a percentage between 0-100 signals sets the shared memory carveout 
                                                               preference in percent of the total shared memory for that kernel launch. 
                                                               This attribute takes precedence over ::cudaFuncAttributePreferredSharedMemoryCarveout.
                                                               This is only a hint, and the driver can choose a different configuration if
                                                               required for the launch.*/  
  , cudaLaunchAttributeNvlinkUtilCentricScheduling   = 16 /**< Valid for streams, graph nodes, launches. This attribute is a hint to the CUDA runtime that the 
                                                                 launch should attempt to make the kernel maximize its NVLINK utilization.  
                                                                 <br>
                                                                 When possible to honor this hint, CUDA will assume each block in the grid launch will carry out an even amount 
                                                                 of NVLINK traffic, and make a best-effort attempt to adjust the kernel launch based on that assumption.
                                                                 <br>
                                                                 This attribute is a hint only. CUDA makes no functional or performance guarantee. Its applicability can be 
                                                                 affected by many different factors, including driver version (i.e. CUDA doesn't guarantee the performance 
                                                                 characteristics will be maintained between driver versions or a driver update could alter or regress 
                                                                 previously observed perf characteristics.) It also doesn't guarantee a successful result, i.e. applying 
                                                                 the attribute may not improve the performance of either the targeted kernel or the encapsulating application.
                                                                 <br>
                                                                 Valid values for ::cudaLaunchAttributeValue::nvlinkUtilCentricScheduling are 0 (disabled) and 1 (enabled).  */
} cudaLaunchAttributeID;

/**
 * Launch attributes union; used as value field of ::cudaLaunchAttribute
 */
typedef __device_builtin__ union cudaLaunchAttributeValue {
    char pad[64]; /* Pad to 64 bytes */
    struct cudaAccessPolicyWindow accessPolicyWindow; /**< Value of launch attribute ::cudaLaunchAttributeAccessPolicyWindow. */
    int cooperative; /**< Value of launch attribute ::cudaLaunchAttributeCooperative. Nonzero indicates a cooperative
                        kernel (see ::cudaLaunchCooperativeKernel). */
    enum cudaSynchronizationPolicy syncPolicy; /**< Value of launch attribute
                                                  ::cudaLaunchAttributeSynchronizationPolicy. ::cudaSynchronizationPolicy
                                                  for work queued up in this stream. */
    /**
     * Value of launch attribute ::cudaLaunchAttributeClusterDimension that
     * represents the desired cluster dimensions for the kernel. Opaque type
     * with the following fields:
     *     - \p x - The X dimension of the cluster, in blocks. Must be a divisor
     *              of the grid X dimension.
     *     - \p y - The Y dimension of the cluster, in blocks. Must be a divisor
     *              of the grid Y dimension.
     *     - \p z - The Z dimension of the cluster, in blocks. Must be a divisor
     *              of the grid Z dimension.
     */
    struct {
        unsigned int x;
        unsigned int y;
        unsigned int z;
    } clusterDim;
    enum cudaClusterSchedulingPolicy clusterSchedulingPolicyPreference; /**< Value of launch attribute
                                                                           ::cudaLaunchAttributeClusterSchedulingPolicyPreference. Cluster
                                                                           scheduling policy preference for the kernel. */
    int programmaticStreamSerializationAllowed; /**< Value of launch attribute
                                                   ::cudaLaunchAttributeProgrammaticStreamSerialization. */

    /**
     * Value of launch attribute ::cudaLaunchAttributeProgrammaticEvent
     * with the following fields:
     *     - \p cudaEvent_t event - Event to fire when all blocks trigger it.
     *     - \p int flags;        - Event record flags, see ::cudaEventRecordWithFlags. Does not accept
     *                               ::cudaEventRecordExternal.
     *     - \p int triggerAtBlockStart - If this is set to non-0, each block launch will automatically trigger the event.
     */
    struct {
        cudaEvent_t event;
        int flags;
        int triggerAtBlockStart;
    } programmaticEvent;
    int priority; /**< Value of launch attribute ::cudaLaunchAttributePriority. Execution priority of the kernel. */
    cudaLaunchMemSyncDomainMap memSyncDomainMap; /**< Value of launch attribute
                                                    ::cudaLaunchAttributeMemSyncDomainMap. See
                                                    ::cudaLaunchMemSyncDomainMap. */
    cudaLaunchMemSyncDomain memSyncDomain;       /**< Value of launch attribute ::cudaLaunchAttributeMemSyncDomain. See
                                                    ::cudaLaunchMemSyncDomain. */
    /**
     * Value of launch attribute ::cudaLaunchAttributePreferredClusterDimension
     * that represents the desired preferred cluster dimensions for the kernel.
     * Opaque type with the following fields:
     *     - \p x - The X dimension of the preferred cluster, in blocks. Must be
     *              a divisor of the grid X dimension, and must be a multiple of
     *              the \p x field of ::cudaLaunchAttributeValue::clusterDim.
     *     - \p y - The Y dimension of the preferred cluster, in blocks. Must be
     *              a divisor of the grid Y dimension, and must be a multiple of
     *              the \p y field of ::cudaLaunchAttributeValue::clusterDim.
     *     - \p z - The Z dimension of the preferred cluster, in blocks. Must be
     *              equal to the \p z field of ::cudaLaunchAttributeValue::clusterDim.
     */
    struct {
        unsigned int x;
        unsigned int y;
        unsigned int z;
    } preferredClusterDim;

    /**
     * Value of launch attribute ::cudaLaunchAttributeLaunchCompletionEvent
     * with the following fields:
     *     - \p cudaEvent_t event - Event to fire when the last block launches.
     *     - \p int flags - Event record flags, see ::cudaEventRecordWithFlags. Does not accept
     *                   ::cudaEventRecordExternal.
     */
    struct {
        cudaEvent_t event;
        int flags;
    } launchCompletionEvent;

    /**
     * Value of launch attribute ::cudaLaunchAttributeDeviceUpdatableKernelNode
     * with the following fields:
     *    - \p int deviceUpdatable - Whether or not the resulting kernel node should be device-updatable.
     *    - \p cudaGraphDeviceNode_t devNode - Returns a handle to pass to the various device-side update functions.
     */
    struct {
        int deviceUpdatable;
        cudaGraphDeviceNode_t devNode;
    } deviceUpdatableKernelNode;
    unsigned int sharedMemCarveout; /**< Value of launch attribute ::cudaLaunchAttributePreferredSharedMemoryCarveout. */
    unsigned int nvlinkUtilCentricScheduling; /**< Value of launch attribute ::cudaLaunchAttributeNvlinkUtilCentricScheduling. */

} cudaLaunchAttributeValue;

/**
 * Launch attribute
 */
typedef __device_builtin__ struct cudaLaunchAttribute_st {
    cudaLaunchAttributeID id; /**< Attribute to set */
    char pad[8 - sizeof(cudaLaunchAttributeID)];
    cudaLaunchAttributeValue val; /**< Value of the attribute */
} cudaLaunchAttribute;

/**
 * CUDA extensible launch configuration
 */
typedef __device_builtin__ struct cudaLaunchConfig_st {
    dim3 gridDim;               /**< Grid dimensions */
    dim3 blockDim;              /**< Block dimensions */
    size_t dynamicSmemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    cudaStream_t stream;        /**< Stream identifier */
    cudaLaunchAttribute *attrs; /**< List of attributes; nullable if ::cudaLaunchConfig_t::numAttrs == 0 */
    unsigned int numAttrs;      /**< Number of attributes populated in ::cudaLaunchConfig_t::attrs */
} cudaLaunchConfig_t;

#define cudaStreamAttrID cudaLaunchAttributeID
#define cudaStreamAttributeAccessPolicyWindow    cudaLaunchAttributeAccessPolicyWindow
#define cudaStreamAttributeSynchronizationPolicy cudaLaunchAttributeSynchronizationPolicy
#define cudaStreamAttributeMemSyncDomainMap      cudaLaunchAttributeMemSyncDomainMap
#define cudaStreamAttributeMemSyncDomain         cudaLaunchAttributeMemSyncDomain
#define cudaStreamAttributePriority cudaLaunchAttributePriority

#define cudaStreamAttrValue cudaLaunchAttributeValue

#define cudaKernelNodeAttrID cudaLaunchAttributeID
#define cudaKernelNodeAttributeAccessPolicyWindow cudaLaunchAttributeAccessPolicyWindow
#define cudaKernelNodeAttributeCooperative        cudaLaunchAttributeCooperative
#define cudaKernelNodeAttributePriority           cudaLaunchAttributePriority
#define cudaKernelNodeAttributeClusterDimension                     cudaLaunchAttributeClusterDimension
#define cudaKernelNodeAttributeClusterSchedulingPolicyPreference    cudaLaunchAttributeClusterSchedulingPolicyPreference
#define cudaKernelNodeAttributeMemSyncDomainMap   cudaLaunchAttributeMemSyncDomainMap
#define cudaKernelNodeAttributeMemSyncDomain      cudaLaunchAttributeMemSyncDomain
#define cudaKernelNodeAttributePreferredSharedMemoryCarveout cudaLaunchAttributePreferredSharedMemoryCarveout
#define cudaKernelNodeAttributeDeviceUpdatableKernelNode cudaLaunchAttributeDeviceUpdatableKernelNode
#define cudaKernelNodeAttributeNvlinkUtilCentricScheduling cudaLaunchAttributeNvlinkUtilCentricScheduling 


#define cudaKernelNodeAttrValue cudaLaunchAttributeValue

/**
 * CUDA device NUMA config
 */
enum __device_builtin__  cudaDeviceNumaConfig {
    cudaDeviceNumaConfigNone  = 0, /**< The GPU is not a NUMA node */
    cudaDeviceNumaConfigNumaNode, /**< The GPU is a NUMA node, cudaDevAttrNumaId contains its NUMA ID */
};

/**
 * CUDA async callback handle
 */
typedef struct cudaAsyncCallbackEntry* cudaAsyncCallbackHandle_t;

struct cudaAsyncCallbackEntry;

/**
* Types of async notification that can occur
*/
typedef __device_builtin__ enum cudaAsyncNotificationType_enum {
    cudaAsyncNotificationTypeOverBudget = 0x1 /**< Sent when the process has exceeded its device memory budget */
} cudaAsyncNotificationType;

/**
* Information describing an async notification event
*/
typedef __device_builtin__ struct cudaAsyncNotificationInfo
{
    cudaAsyncNotificationType type; /**< The type of notification being sent */
    union {
        struct {
            unsigned long long bytesOverBudget; /**< The number of bytes that the process has allocated above its device memory budget */
        } overBudget; /**< Information about notifications of type \p cudaAsyncNotificationTypeOverBudget */
    } info; /**< Information about the notification. \p type must be checked in order to interpret this field. */
} cudaAsyncNotificationInfo_t;

typedef void (*cudaAsyncCallback)(cudaAsyncNotificationInfo_t*, void*, cudaAsyncCallbackHandle_t);

typedef __device_builtin__ enum CUDAlogLevel_enum {
    cudaLogLevelError = 0,
    cudaLogLevelWarning = 1
} cudaLogLevel;

typedef __device_builtin__ struct CUlogsCallbackEntry_st *cudaLogsCallbackHandle;
typedef __device_builtin__ unsigned int cudaLogIterator;

/** @} */
/** @} */ /* END CUDART_TYPES */

#endif  /* !__CUDACC_RTC_MINIMAL__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_DRIVER_TYPES_H__
#endif

#undef __CUDA_DEPRECATED



#endif /* !__DRIVER_TYPES_H__ */
