//
// Created by salmon on 18-4-18.
//

#include "DAGExecutor.h"
namespace simpla {

struct DAGExecutorImpl : public DAGExecutor {
    void RunAsync(DoneCallback done) override;
};
void DAGExecutorImpl::RunAsync(simpla::DAGExecutor::DoneCallback done) {}
}  // namespace simpla {
