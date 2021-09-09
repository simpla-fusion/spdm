//
// Created by salmon on 18-4-18.
//

#ifndef SIMPLA_DAGEXECUTOR_H
#define SIMPLA_DAGEXECUTOR_H

#include <functional>
#include "SpDM.h"
namespace sp {
struct Notification {
    void Notify();
    void WaitForNotification();
};
struct Status;
struct DAGExecutor {
    virtual ~DAGExecutor() = default;
    typedef std::function<void(const Status&)> DoneCallback;

    virtual void RunAsync(DoneCallback done) = 0;

    // Synchronous wrapper for RunAsync().
    template <typename... Args>
    int Run(const Args&&... args) {
        Notification n;
        RunAsync([&n](const Status& s) { n.Notify(); }, std::forward<Args>(args)...);
        n.WaitForNotification();
        return kSuccessful;
    }
};
}  // namespace sp
#endif  // SIMPLA_DAGEXECUTOR_H
