#ifndef SP_DOCUMENTBACKEND_H_
#define SP_DOCUMENTBACKEND_H_
#include "SpDocument.h"
#include <any>

namespace sp
{

    class SpNode::Backend
    {
    public:
        Backend();
        virtual ~Backend();
        virtual void set_attribute(std::string const &name, std::any const &v) = 0;
        virtual std::any get_attribute(std::string const &name) = 0;
        virtual void remove_attribute(std::string const &name) = 0;
    };
} // namespace sp

#endif //SP_DOCUMENTBACKEND_H_