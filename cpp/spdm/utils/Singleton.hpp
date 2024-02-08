/**
 * Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * @file singleton_holder.h
 *
 *  created on: 2008-04-16
 *      Author: salmon
 */
#ifndef SP_SINGLETON_H_
#define SP_SINGLETON_H_
namespace sp
{

    /** @ingroup design_pattern
     *
     * @addtogroup  singleton Singleton
     * @{
     *
     * @brief singleton
     *
     * @note  Meyers Singleton，
     * Ref:Andrei Alexandrescu Chap 6.4
     * Modern C++ Design Generic Programming and Design Patterns Applied 2001 Addison Wesley ,
     */
    template <class T>
    class Singleton
    {
    public:
        static T &instance()
        {
            if (!pInstance_)
            {
                //#pragma omp critical
                // TOD add some for mt critical
                if (!pInstance_)
                {
                    static T tmp;
                    pInstance_ = &tmp;
                }
            }
            return *pInstance_;
        }

    protected:
        Singleton() {}
        ~Singleton() {}
        static T *volatile pInstance_;
    };

    template <class T>
    T *volatile Singleton<T>::pInstance_ = 0;

    /** @} */
} // namespace sp
#endif // SP_SINGLETON_H_
