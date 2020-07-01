#ifndef SPOID_H_
#define SPOID_H_

class SpOID
{
public:
    SpOID();

    int id() const { return m_id_; }

private:
    int m_id_ = 0;
};

#endif //SPOID_H_