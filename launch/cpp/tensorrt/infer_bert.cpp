/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bert_infer.h"
#include <string>
#include <array>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string enginePath = "/data/workspace/coastcao/tools/TensorRT/demo/BERT/engines/bert_large_128.engine"; //"engines/coqui_vits_bs1_sl128.engine";
    int batchSize = 1;
    int seqLen = 128;

    BertInference bert(enginePath, batchSize, seqLen, true);
    std::cout << "[0]bert.mDeviceBuffers.size():" << bert.mDeviceBuffers.size() << std::endl;
    bert.prepare(0, batchSize);
    std::cout << "[1]bert.mDeviceBuffers.size():" << bert.mDeviceBuffers.size() << std::endl;


    int inputIds[seqLen] = {101, 2054, 2003, 23435, 5339, 1029, 102, 23435, 5339,
                            2003, 1037, 2152, 2836, 2784, 4083, 28937, 4132, 2008,
                            18058, 2659, 2397, 9407, 1998, 2152, 2083, 18780, 2005,
                            18726, 2107, 2004, 16755, 2545, 1010, 4613, 1998, 3746,
                            1013, 2678, 2006, 1050, 17258, 2401, 14246, 2271, 1012,
                            2009, 2950, 11968, 8043, 2015, 2000, 12324, 4275, 1010,
                            1998, 13354, 7076, 2000, 2490, 3117, 23092, 1998, 9014,
                            2077, 11243, 20600, 2015, 2005, 28937, 1012, 2651, 1050,
                            17258, 2401, 2003, 2330, 1011, 14768, 6129, 11968, 8043,
                            2015, 1998, 13354, 7076, 1999, 23435, 5339, 2061, 2008,
                            1996, 2784, 4083, 2451, 2064, 7661, 4697, 1998, 7949,
                            2122, 6177, 2000, 2202, 5056, 1997, 3928, 23435, 5339,
                            20600, 2015, 2005, 2115, 18726, 1012, 102, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0};
    int segmentIds[seqLen] = {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
                            ,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
                            ,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
                            ,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int inputMask[seqLen] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
                            ,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
                            ,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
                            ,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0};

    //const void* inputIdsPtr = inputIds.ptr;
    //const void* segmentIdsPtr = segmentIds.request().ptr;
    //const void* inputMaskPtr = inputMask.request().ptr;
    std::cout << "[2]bert.mDeviceBuffers.size():" << bert.mDeviceBuffers.size() << std::endl;
    bert.run(&inputIds, &segmentIds, &inputMask, 0, 1);

    //auto output = py::array_t<float>(bert.mOutputDims, (float*) bert.mHostOutput.data());
    auto output = (float*) bert.mHostOutput.data();
    auto output_dims = bert.mOutputDims;
    for(auto x: output_dims) {
        std::cout << "output_dim:" << x << std::endl;
    }

    /*const size_t numOutputItems = batchSize * seqLen * 2;
    for (int x = 0; x < numOutputItems; x++)
        std::cout << output[x] << std::endl;*/
   
}
