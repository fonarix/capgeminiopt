
#ifndef __TaskCuda3Utils_h__
#define __TaskCuda3Utils_h__

#include <string>
#include <sstream>
#include <iomanip>

template<class T>
std::stringstream& PrintCell(std::stringstream& ss, std::size_t padding, const T& value)
{
    ss << "|" << std::setw(padding) << value;
    return ss;
}


struct ReduceFuncResult
{
    //using value_type = double;
    using value_type = float;

    std::string mName;
    int mThreadsPerBlock;
    int mReduceMultiplier;

    ReduceFuncResult(std::string inName, int inThreadsPerBlock, int inReduceMultiplier)
        : mName(std::move(inName))
        , mThreadsPerBlock(inThreadsPerBlock)
        , mReduceMultiplier(inReduceMultiplier)
        , mMallocTimeMs(0.0)
        , mCopyHostToDeviceTimeMs(0.0)
        , mReduceTimeMs(0.0)
        , mCopyDeviceToHostTimeMs(0.0)
        , mFreeTimeMs(0.0)
        , mTotalTime(0.0)
        , mReduceResult(0.0)
    {
    }

    value_type mMallocTimeMs;
    value_type mCopyHostToDeviceTimeMs;
    value_type mReduceTimeMs;
    value_type mCopyDeviceToHostTimeMs;
    value_type mFreeTimeMs;
    value_type mTotalTime;
    value_type mReduceResult;

    static std::string ToStringHeader()
    {
        std::stringstream ss;
        constexpr std::size_t PDD = 12;

        ss << std::fixed << std::showpoint;
        ss << std::setfill(' ') << std::setw(0);

        PrintCell(ss,   7, "Kernel");
        PrintCell(ss,   4, "Mult");
        PrintCell(ss,   8, "ThPerBlk");
        PrintCell(ss,   9, "Malloc");
        PrintCell(ss, PDD, "CpHstToDev");
        PrintCell(ss,   9, "Reduce");
        PrintCell(ss, PDD, "CpDevToHst");
        PrintCell(ss,   9, "Free");
        PrintCell(ss, PDD, "TotalTime");
        PrintCell(ss,  18, "ReduceRes");
        ss << "|";

        return ss.str();
    }

    std::string ToString() const
    {
        std::stringstream ss;
        constexpr std::size_t PDD = 12;

        ss << std::fixed << std::showpoint;
        ss << std::setfill(' ') << std::setw(0) << std::setprecision(4);

        PrintCell(ss,   7, mName);
        PrintCell(ss,   4, mReduceMultiplier);
        PrintCell(ss,   8, mThreadsPerBlock);
        PrintCell(ss,   9, mMallocTimeMs);
        PrintCell(ss, PDD, mCopyHostToDeviceTimeMs);
        PrintCell(ss,   9, mReduceTimeMs);
        PrintCell(ss, PDD, mCopyDeviceToHostTimeMs);
        PrintCell(ss,   9, mFreeTimeMs);
        PrintCell(ss, PDD, mTotalTime);
        PrintCell(ss, 18 , mReduceResult);
        ss << "|";

        return ss.str();
    }
};

#endif // #ifndef __TaskCuda3Utils_h__