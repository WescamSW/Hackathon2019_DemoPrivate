/*
 * api.cpp
 *
 * Create on March 27, 2019
 *
 */

Api::Api(int numberOfDrones)
{
    init(numberOfDrones);

    for (int i = 0; i < g_drones.size(); i++) {
        startDrone(i);
    }

}



void Api::takeoffDrone(int droneId) {

    ControlPtr       control = g_drones[droneId]->getDroneController();
    CameraControlPtr camera  = g_drones[droneId]->getCameraControl();
    PilotPtr         pilot   = g_drones[droneId]->getPilot();
    VideoDriverPtr   video   = g_drones[droneId]->getVideoDriver();
    g_drones[droneId]->getPilot()->takeOff();
    waitSeconds(3);

    if (droneId == 1) {
        setFlightAltitude(droneId, 1.5f)
    } else if (droneId == 2) {
        setFlightAltitude(droneId, 2.5f)
    } else if (droneId == 3) {
        setFlightAltitude(droneId, 3.5f)
    } else if (droneId == 4) {
        setFlightAltitude(droneId, 4.5f)
    }
}

void Api::landDrone(int droneId)
{
    g_drones[droneId]->getCameraControl()->setForward();
    g_drones[droneId]->getPilot()->land();
    waitSeconds(5);
}

void Api::stopDrone(int droneId)
{
    landDrone(droneId);
    g_drones[droneId]->getVideoDriver()->stop();
    g_drones[droneId]->getDroneController()->stop();
}

/*********************/
/* PRIVATE FUCNTIONS */
/*********************/
Api::init(int numberOfDrones)
{
    if (numberOfDrones < 1 || numberOfDrones > 4) {
        cout << "\nYou must specify a number of drones between 1 and 3" << endl;
        exit(EXIT_SUCCESS);
    }

    if (numberOfDrones >= 1) {
        g_frames.emplace_back(make_shared<VideoFrameGeneric>(BEBOP2_STREAM_HEIGHT, BEBOP2_STREAM_WIDTH));
        g_drones.emplace_back(make_shared<Bebop2>(Callsign::ALPHA, g_frames[0]));
    }

    if (numberOfDrones >= 2) {
        g_frames.emplace_back(make_shared<VideoFrameGeneric>(BEBOP2_STREAM_HEIGHT, BEBOP2_STREAM_WIDTH));
        g_drones.emplace_back(make_shared<Bebop2>(Callsign::LONE_WOLF, g_frames[1]));
    }

    if (numberOfDrones >= 3) {
        g_frames.emplace_back(make_shared<VideoFrameGeneric>(BEBOP2_STREAM_HEIGHT, BEBOP2_STREAM_WIDTH));
        g_drones.emplace_back(make_shared<Bebop2>(Callsign::BRAVO, g_frames[2]));
    }

    if (numberOfDrones >= 4) {
        g_frames.emplace_back(make_shared<VideoFrameGeneric>(BEBOP2_STREAM_HEIGHT, BEBOP2_STREAM_WIDTH));
        g_drones.emplace_back(make_shared<Bebop2>(Callsign::CHARLIE, g_frames[3]));
    }
}

void Api::startDrone(int droneId)
{

    ControlPtr       control = g_drones[droneId]->getDroneController();
    CameraControlPtr camera  = g_drones[droneId]->getCameraControl();
    PilotPtr         pilot   = g_drones[droneId]->getPilot();
    VideoDriverPtr   video   = g_drones[droneId]->getVideoDriver();

    control->start();
    cout << "Done CONTROL START" << endl;
    waitSeconds(1);
    video->start();
    cout << "Done VIDEO START" << endl;
    camera->setForward();
    waitSeconds(1);
    cout << "Done drone start" << endl;
}

void Api::setFlightAltitude(int droneId, float heightMeters)
{
    PilotPtr         pilot   = g_drones[droneId]->getPilot();
    const int WAIT_TIME = WAIT_MED;
    waitSeconds(WAIT_SHORT);
    pilot->moveRelativeMetres(0.0, 0.0, -heightMeters); waitSeconds(WAIT_TIME);
}

/********
* API
- Pilot
    - takeoff?
    - land()?

    - moveRelativeMetres()
    - moveDirection()?
    - setHeading()

- DroneController <- not directly used
    - start()
    - stop()

- CameraController
    - setTiltPan()
    - setForward()
    - capturePhoto()

-
*/