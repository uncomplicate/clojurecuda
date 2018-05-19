//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

package uncomplicate.clojurecuda.internal;

import jcuda.driver.JCudaDriver;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUresult;

public class SafeCUdeviceptr extends CUdeviceptr implements CUReleaseable {

    private boolean released = false;

    public boolean isReleased () {
        return released;
    }

    public synchronized int release () {
        if (!released) {
            int status = -Integer.MIN_VALUE;
            try {
                status = JCudaDriver.cuMemFree(this);
            } finally {
                released = (CUresult.CUDA_SUCCESS == status);
                return status;
            }
        }
        return CUresult.CUDA_SUCCESS;
    }

}
