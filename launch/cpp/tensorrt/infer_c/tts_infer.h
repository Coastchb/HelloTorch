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

#ifndef INFER_C_TTS_INFER_H
#define INFER_C_TTS_INFER_H

#include "common.h"
#include "logging.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <numeric>
#include <string.h>
#include <vector>

using namespace nvinfer1;

struct TTSInference
{
    TTSInference(
        const std::string& enginePath, const int maxBatchSize, const int seqLength, const bool enableGraph = false)
        : mSeqLength(seqLength)
        , mEnableGraph(enableGraph)
    {
        gLogInfo << "--------------------\n";
        gLogInfo << "Using TTS inference C++\n";
        if (enableGraph)
        {
            gLogInfo << "CUDA Graph is enabled\n";
        }
        else
        {
            gLogInfo << "CUDA Graph is disabled\n";
        }

        gLogInfo << "--------------------\n";

        initLibNvInferPlugins(&gLogger, "");

        gLogInfo << "Loading TTS Inference Engine ... \n";
        std::ifstream input(enginePath, std::ios::binary);
        if (!input)
        {
            gLogError << "Error opening engine file: " << enginePath << "\n";
            exit(-1);
        }

        input.seekg(0, input.end);
        const size_t fsize = input.tellg();
        input.seekg(0, input.beg);

        std::vector<char> bytes(fsize);
        input.read(bytes.data(), fsize);
        gLogInfo << "fsize:" << fsize << "\n";

        mRuntime = TrtUniquePtr<IRuntime>(createInferRuntime(gLogger));
        if (mRuntime == nullptr)
        {
            gLogError << "Error creating TRT mRuntime\n";
            exit(-1);
        }

        mEngine = TrtUniquePtr<ICudaEngine>(mRuntime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (mEngine == nullptr)
        {
            gLogError << "Error deserializing CUDA engine\n";
            exit(-1);
        }
        gLogInfo << "Done\n";

        mEnableVariableLen = mEngine->getNbIOTensors() == kVITS_INPUT_NUM + 1 ? false : true;
        if (mEnableVariableLen)
        {
            gLogInfo << "Variable length is enabled\n";
        }
        else
        {
            gLogInfo << "Variable length is disabled\n";
        }

        mContext = TrtUniquePtr<IExecutionContext>(mEngine->createExecutionContext());
        if (!mContext)
        {
            gLogError << "Error creating execution context\n";
            exit(-1);
        }

        gpuErrChk(cudaStreamCreate(&mStream));

        allocateBindings(maxBatchSize);
    }

    void allocateBindings(const int maxBatchSize)
    {
        gLogInfo << "allocateBindings start\n";

        for (int i = 0; i < kVITS_INPUT_NUM; i++)
        {
            void* devBuf;
            size_t allocationSize = 0;
            if (i == 0)
                allocationSize = mSeqLength * maxBatchSize * sizeof(int64_t);
            else if (i == 1)
                allocationSize = maxBatchSize * sizeof(int64_t);
            else if (i == 2)
                allocationSize = 3 * sizeof(float);
            else if (i == 3)
                allocationSize = 1 * sizeof(int64_t);                                
            gpuErrChk(cudaMalloc(&devBuf, allocationSize));
            gpuErrChk(cudaMemset(devBuf, 0, allocationSize));
            mDeviceBuffers.emplace_back(devBuf);
            mInputSizes.emplace_back(allocationSize);
        }

        // const size_t numOutputDIMs = maxBatchSize * mSeqLength * 2;
        const size_t numOutputDIMs = 44544; //maxBatchSize * mSeqLength;
        mOutputSize = numOutputDIMs * sizeof(float);
        if (mEnableVariableLen)
        {
            mOutputDims = {maxBatchSize * mSeqLength * 2};
        }
        else
        {
            mOutputDims = {maxBatchSize, 1, numOutputDIMs};
        }
        void* devBuf;
        gpuErrChk(cudaMalloc(&devBuf, mOutputSize));
        gpuErrChk(cudaMemset(devBuf, 0, mOutputSize));
        mDeviceBuffers.emplace_back(devBuf);
        mHostOutput.resize(numOutputDIMs);
        gLogInfo << "allocateBindings done\n";
    }

    void prepare(int profIdx, int batchSize)
    {
        gLogInfo << "prepare start\n";
        mContext->setOptimizationProfileAsync(profIdx, mStream);

        if (mEnableVariableLen)
        {
            const int allocationSizes[] = {mSeqLength * batchSize, mSeqLength * batchSize, batchSize + 1, mSeqLength};
            for (int i = 0; i < sizeof(allocationSizes)/sizeof(allocationSizes[0]); i++)
            {
                auto const tensorName = mEngine->getIOTensorName(i % mEngine->getNbIOTensors());
                mContext->setInputShape(tensorName, Dims{1, {allocationSizes[i]}});
            }
        }
        else
        {
            gLogInfo << "not mEnableVariableLen\n";
            for (int i = 0; i < kVITS_INPUT_NUM; i++)
            {
                auto const tensorName = mEngine->getIOTensorName(i);
                gLogInfo << "tensorName:" << tensorName << "\n";
                auto dims = mContext->getTensorShape(tensorName).d;
                for (int x=0; x < 3; x++)
                    gLogInfo << "[0]Tensor shape: " << dims[x] << "\t";
                gLogInfo << "\n";
                //gLogInfo << "Tensor shape: " << dims.d << ", " << dims.d‌:ml-citation{ref="1" data="citationList"} << ", " << dims.d‌:ml-citation{ref="2" data="citationList"} << "\n";
                
                if (i == 0)
                    mContext->setInputShape(tensorName, Dims2(batchSize, mSeqLength));
                else if (i == 1)
                    mContext->setInputShape(tensorName, Dims{1, {batchSize}});
                else if (i == 2)
                    mContext->setInputShape(tensorName, Dims{1, {3}});
                else if (i == 3)
                    mContext->setInputShape(tensorName, Dims{1, {1}});

                gLogInfo << "tensorName:" << tensorName << "\n";
                auto dims1 = mContext->getTensorShape(tensorName).d;
                for (int x=0; x < 3; x++)
                    gLogInfo << "[1]Tensor shape: " << dims1[x] << "\t";
                gLogInfo << "\n";
            }
        }

        if (!mContext->allInputDimensionsSpecified())
        {
            gLogError << "Not all input dimensions are specified for the exeuction context\n";
            exit(-1);
        }
        gLogInfo << "setInputShape done\n";
        if (mEnableGraph)
        {
            for (int32_t i = 0; i < mEngine->getNbIOTensors(); i++)
            {
                gLogInfo << "setTensorAddress 。。。\n";
                auto const& name = mEngine->getIOTensorName(i);
                mContext->setTensorAddress(name, mDeviceBuffers[i]);
            }
            gLogInfo << "setTensorAddress done\n";

            cudaGraph_t graph;
            cudaGraphExec_t exec;
            // warm up and let mContext do cublas initialization
            bool status = mContext->enqueueV3(mStream);
            if (!status)
            {
                gLogError << "[0]Enqueue failed\n";
                exit(-1);
            }
            gLogVerbose << "Capturing graph\n";
            
            gpuErrChk(cudaStreamBeginCapture(mStream, cudaStreamCaptureModeRelaxed));
            status = mContext->enqueueV3(mStream);
            if (!status)
            {
                gLogError << "[1]Enqueue failed\n";
                exit(-1);
            }

            gpuErrChk(cudaStreamEndCapture(mStream, &graph));
            gpuErrChk(cudaStreamSynchronize(mStream));

            gpuErrChk(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
            mExecGraph = exec;
        }
        mCuSeqlens.resize(batchSize + 1);
        std::generate(mCuSeqlens.begin(), mCuSeqlens.end(), [pos = -mSeqLength, this]() mutable{ pos += mSeqLength; return pos; });

        gLogInfo << "prepare done\n";
    }

    void run(const void* const* inputBuffers, int warmUps, int iterations)
    {
        for (int i = 0; i < kVITS_INPUT_NUM; i++)
        {
            gpuErrChk(
                cudaMemcpyAsync(mDeviceBuffers[i], inputBuffers[i], mInputSizes[i], cudaMemcpyHostToDevice, mStream));
        }

        gLogInfo << "Warming up " << warmUps << " iterations ...\n";
        for (int it = 0; it < warmUps; it++)
        {
            if (mEnableGraph)
            {
                gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
            }
            else
            {
                bool status = mContext->enqueueV3(mStream);
                if (!status)
                {
                    gLogError << "Enqueue failed\n";
                    exit(-1);
                }
            }
        }
        gpuErrChk(cudaStreamSynchronize(mStream));

        cudaEvent_t start, stop;
        gpuErrChk(cudaEventCreate(&start));
        gpuErrChk(cudaEventCreate(&stop));

        std::vector<float> times;
        gLogInfo << "Running " << iterations << " iterations ...\n";
        for (int it = 0; it < iterations; it++)
        {
            gpuErrChk(cudaEventRecord(start, mStream));
            if (mEnableGraph)
            {
                gpuErrChk(cudaGraphLaunch(mExecGraph, mStream));
            }
            else
            {
                bool status = mContext->enqueueV3(mStream);
                if (!status)
                {
                    gLogError << "Enqueue failed\n";
                    exit(-1);
                }
            }
            gpuErrChk(cudaEventRecord(stop, mStream));
            gpuErrChk(cudaStreamSynchronize(mStream));
            float time;
            gpuErrChk(cudaEventElapsedTime(&time, start, stop));
            times.push_back(time);
        }

        gpuErrChk(cudaMemcpyAsync(
            mHostOutput.data(), mDeviceBuffers[mEnableVariableLen ? kVITS_INPUT_NUM + 1 : kVITS_INPUT_NUM], mOutputSize, cudaMemcpyDeviceToHost, mStream));

        gpuErrChk(cudaStreamSynchronize(mStream));

        mTimes.push_back(times);
    }

    void run(const void* inputIds, const void* segmentIds, const void* inputMask, int warmUps, int iterations)
    {
        if (mEnableVariableLen)
        {
            const std::vector<const void*> inputBuffers = {inputIds, segmentIds, mCuSeqlens.data()};
            run(inputBuffers.data(), warmUps, iterations);
        }
        else
        {
            const std::vector<const void*> inputBuffers = {inputIds, segmentIds, inputMask};
            run(inputBuffers.data(), warmUps, iterations);
        }
    }

    void run(int profIdx, int batchSize, const void* inputIds, const void* segmentIds, const void* inputMask,
        int warmUps, int iterations)
    {

        prepare(profIdx, batchSize);
        run(inputIds, segmentIds, inputMask, warmUps, iterations);
    }

    void reportTiming(int batchIndex, int batchSize)
    {

        std::vector<float>& times = mTimes[batchIndex];
        const float totalTime = std::accumulate(times.begin(), times.end(), 0.0);
        const float avgTime = totalTime / times.size();

        sort(times.begin(), times.end());
        const float percentile95 = times[(int) ((float) times.size() * 0.95)];
        const float percentile99 = times[(int) ((float) times.size() * 0.99)];
        const int throughput = (int) ((float) batchSize * (1000.0 / avgTime));
        gLogInfo << "Running " << times.size() << " iterations with Batch Size: " << batchSize << "\n";
        gLogInfo << "\tTotal Time: " << totalTime << " ms \n";
        gLogInfo << "\tAverage Time: " << avgTime << " ms\n";
        gLogInfo << "\t95th Percentile Time: " << percentile95 << " ms\n";
        gLogInfo << "\t99th Percentile Time: " << percentile99 << " ms\n";
        gLogInfo << "\tThroughput: " << throughput << " sentences/s\n";
    }

    ~TTSInference()
    {

        gpuErrChk(cudaStreamDestroy(mStream));

        for (auto& buf : mDeviceBuffers)
        {
            gpuErrChk(cudaFree(buf));
        }
    }

    static const int kVITS_INPUT_NUM = 4;

    const int mSeqLength;
    const bool mEnableGraph;

    TrtUniquePtr<IRuntime> mRuntime{nullptr};
    TrtUniquePtr<ICudaEngine> mEngine{nullptr};
    TrtUniquePtr<IExecutionContext> mContext{nullptr};
    bool mEnableVariableLen;
    std::vector<int> mCuSeqlens;

    cudaStream_t mStream{NULL};
    std::vector<void*> mDeviceBuffers;
    std::vector<float> mHostOutput;
    std::vector<size_t> mInputSizes;
    size_t mOutputSize;
    std::vector<int> mOutputDims;

    std::vector<std::vector<float>> mTimes;

    cudaGraphExec_t mExecGraph;
};

#endif // INFER_C_TTS_INFER_H
