#ifndef PY_BEBOP_H_
#define PY_BEBOP_H_

#include <vector>
class Api {
public:
    /* TODO what does setting = delete do? */

    /// Construct a number of Bebop 2 drone instances
    /// @param numDroneInstances number of drone instances to create (valid range 1-3)
    Api(int numOfDrones);

    /// Default destructor
    ~Api() {}

    int count[2];

private:
    void initDrones(int numberOfDrones);
    void startDrone(int droneId);
    void setFlightAltitude(int droneId, float heightMeters);

    vector<shared_ptr<Bebop2>>            g_drones;
    vector<shared_ptr<VideoFrameGeneric>> g_frames;
}
#endif /* PY_BEBOP_H_ */