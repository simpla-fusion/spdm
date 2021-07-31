//
// Created by salmon on 18-4-18.
//

#include "DAGExecutor.h"
namespace sp {

struct DAGExecutorImpl : public DAGExecutor {
    void RunAsync(DoneCallback done) override;
};
void DAGExecutorImpl::RunAsync(sp::DAGExecutor::DoneCallback done) {}
}  // namespace sp {
