#ifndef API_H_
#define API_H_

#include <vector>
class Api {
public:
    /* TODO what does setting = delete do? */

    /// Construct a number of Bebop 2 drone instances
    /// @param numDroneInstances number of drone instances to create (valid range 1-3)
    Api(int numOfDrones);

    /// Default destructor
    ~Api() {}

private:
    void initDrones(int numberOfDrones);
    void startDrone(int droneId);
    void setFlightAltitude(int droneId, float heightMeters);

    vector<shared_ptr<Bebop2>>            g_drones;
    vector<shared_ptr<VideoFrameGeneric>> g_frames;
}
#endif /* API_H_ */