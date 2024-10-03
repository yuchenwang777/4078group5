import time
import random

def locate_update(self):

    self.take_pic()
    meas = self.control()

    self.update_slam(meas)

    return self.ekf.robot.state

def find_my_location(self):
        
        got_two = False

        rand = random.choice([-1, 1])

        start_time = time.time()

        while not got_two:

            self.locate_update()

            lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)

            i = 0

            condi = False
            for lm in lms:
                if lm.tag in self.ekf.taglist:
                    i+=1
                    if i >1:
                        condi = True

            time.sleep(0.2)

            if condi:
                self.pibot.set_velocity([0, 0])

                self.locate_update()

                got_two = True

            else:
                self.pibot.set_velocity([0, rand], turning_tick=self.turn_vel,time=0.2)

            passed_time = time.time() - start_time 

            if passed_time >= 45:  
             break
        
                        