import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class StrikerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Pitch size
        self.pitchwidth = 10.0
        self.pitchheight = 6.0
        #added below constants to support robot complexity
        self.wheelseparation = 0.52
        self.maxlinearvelocity = 1.2 #was moving too fast played with the values here lol
        self.maxangularvelocity = 2.0
        #Wheel actions = [left wheel torque, right wheel torque]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(         #observing the sensors in based space for [distance_to_ball, angle_to_ball,distance_to_goal, angle_to_goal,distance_to_defender, angle_to_defender, striker_velocity, striker_angularvelocity]
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
        self.dt = 0.1
        self.maxsteps = 500

        #pygame animation setup
        self.rendermode = None
        self.window = None
        self.clock = None
        self.renderscalefactor = 80 #im using this cause the width and heights ive defined earlier are real world, this adjusts my values to be screen-rendering friendly

        self.reset()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.strikerposition = np.array([ #at the start of the game
            self.np_random.uniform(0.5, 3.0),
            self.np_random.uniform(1.0, 5.0)
        ], dtype=np.float32)
        self.ballposition = np.array([ #starting the game position for ball
            self.np_random.uniform(3.0, 6.0),
            self.np_random.uniform(1.5, 4.5)
        ], dtype=np.float32)
        self.strikerangle = 0.0
        self.strikervelocity = 0.0
        self.strikerangularvelocity = 0.0

        self.goalposition = np.array([9.5, 3.0], dtype=np.float32) #goal position at starting state
        self.defenderposition = np.array([
            8.5,
            np.random.uniform(2.0, 4.0)
        ], dtype=np.float32)  #defender starting position
        self.defenderangle = np.pi / 2  #defender starts facing vertically
        self.defenderdirection = 1  #used to move defender up and down
        self.previousdistancetoball = self.distance( #for reward/RL memory
            self.strikerposition,
            self.ballposition
        )
        self.previousballtogoal = self.distance( #for reward/RL memory
            self.ballposition,
            self.goalposition
        )
        sensorvector = self.getsensorvalues()
        info = {}
        return sensorvector, info

    def step(self, movementcontroller):
        self.steps += 1
        leftwheel, rightwheel = np.clip(movementcontroller, -1.0, 1.0)
        leftspeed = leftwheel * self.maxlinearvelocity
        rightspeed = rightwheel * self.maxlinearvelocity
        forwardspeed = (leftspeed + rightspeed) / 2.0                   #for differential drive calculations
        turningspeed = (rightspeed - leftspeed) / self.wheelseparation  #for differential drive calculations
        turningspeed = np.clip( #added turningspeed physics for striker agent
            turningspeed,
            -self.maxangularvelocity,
            self.maxangularvelocity
        )
        self.strikervelocity = forwardspeed
        self.strikerangularvelocity = turningspeed
        self.strikerangle += turningspeed * self.dt #to change/update the strikers rotation

        direction = np.array([ #direction for forward vector
            np.cos(self.strikerangle),
            np.sin(self.strikerangle)
        ], dtype=np.float32)
        reward = self.calculatereward() #CALCULATING MY REWARD FUNCTION

        self.strikerposition = np.clip(  ##stopping the striker to not move thru the walls + terminating with outofbounds (using defensive programming here to avoid glitches)
            self.strikerposition,
            [0.0, 0.0],
            [self.pitchwidth, self.pitchheight]
        )

        self.strikerposition += direction * forwardspeed * self.dt           #this updates the strikers direction
        if self.distance(self.strikerposition, self.ballposition) < 0.35:    #for if ball collides
            self.ballposition += direction * 0.15 #<<< swapped from .25 as it wasnt realistic to hit ball with that power with the defender goalie there
        if self.ballposition[0] < 0 or self.ballposition[0] > self.pitchwidth or self.ballposition[1] < 0 or self.ballposition[1] > self.pitchheight: ##FIX BALL GOING OUT OF BOUNDS
            reward -= 10
            terminated = True


        self.updatedefender()

        terminated = False
        truncated = False

        #POINT ALLOCATIONS FOR RL
        if self.goalscored():
            reward += 100
            terminated = True
        if self.outofbounds():
            reward -= 10
            terminated = True
        if self.defendercollision():
            reward -= 5
            #terminated = True ##<< hashed for now so the robot doesnt panic around the defender
        #set for timeout
        if self.steps >= self.maxsteps:
            truncated = True

        sensorvector = self.getsensorvalues()
        info = {
            "goalscored": self.goalscored(),
            "defendercollision": self.defendercollision(),
            "steps": self.steps
        }
        return sensorvector, reward, terminated, truncated, info

    def getsensorvalues(self):

        distance_to_ball, angle_to_ball = self.relativeinfo(
            self.ballposition
        )
        distance_to_goal, angle_to_goal = self.relativeinfo(
            self.goalposition
        )
        distance_to_defender, angle_to_defender = self.relativeinfo(
            self.defenderposition
        )
        sensorvector = np.array([
            distance_to_ball,
            angle_to_ball,
            distance_to_goal,
            angle_to_goal,
            distance_to_defender,
            angle_to_defender,
            self.strikervelocity,
            self.strikerangularvelocity
        ], dtype=np.float32)

        return sensorvector
    def calculatereward(self):
        reward = 0.0
        currentdistancetoball = self.distance(
            self.strikerposition,
            self.ballposition
        )
        currentballtogoal = self.distance(
            self.ballposition,
            self.goalposition
        )
        reward += (                         #for if the agent moves towards ball
                self.previousdistancetoball
                - currentdistancetoball
        )
        reward += 2.0 * (                   #for moving the ball towards the goal
                self.previousballtogoal
                - currentballtogoal
        )
        reward -= 0.01 #<< time penalty (kept minimal such that effect isnt drastic)
        distance_to_ball, angle_to_ball = self.relativeinfo(
            self.ballposition
        )
        reward += 0.02 * np.cos(angle_to_ball)  #reward for facing towards the ball
        reward -= 0.005 * abs(self.strikerangularvelocity)  #penalty for  spinning too much
        goalcentrey = 3.0

        reward -= 0.02 * abs(
            self.ballposition[1] - goalcentrey
        )  # reward shaping for keeping the ball aligned with the center of the goal
        if self.distance(self.ballposition, self.defenderposition) < 0.6: #penalty for when ball gets too close to defender
            reward -= 0.05
        if self.ballposition[0] > 7.0 and self.distance(self.ballposition, self.defenderposition) > 0.8: #reward to being aligned with goal space instead of just following the defender
            reward += 0.05
        self.previousdistancetoball = currentdistancetoball
        self.previousballtogoal = currentballtogoal
        return float(reward)

    def relativeinfo(self, targetposition):
        vector = targetposition - self.strikerposition
        distance = np.linalg.norm(vector)
        absoluteangle = np.arctan2(
            vector[1],
            vector[0]
        )

        relativeangle = absoluteangle - self.strikerangle
        relativeangle = np.arctan2(
            np.sin(relativeangle), #keeping the angle between negative and positive pi using arctan math through my import of numpy
            np.cos(relativeangle)
        )
        return distance, relativeangle

    def distance(self, position1, position2):
        return np.linalg.norm(position1 - position2)

    def updatedefender(self):
        self.defenderposition[1] += self.defenderdirection * 0.03 #defender speed
        if self.defenderposition[1] > 4.6:
            self.defenderdirection = -1
            self.defenderangle = -np.pi / 2

        if self.defenderposition[1] < 1.4:
            self.defenderdirection = 1
            self.defenderangle = np.pi / 2
        if self.distance(self.defenderposition, self.ballposition) < 0.45:
            if self.defenderposition[1] >= self.ballposition[1]:
                self.defenderposition[1] += 0.08
            else:
                self.defenderposition[1] -= 0.08
    def goalscored(self):
        return (
                self.ballposition[0] >= 9.3
                and
                2.2 <= self.ballposition[1] <= 3.8 #used to set the opening on my pitch (set at 3.8-2.2 = 1.6)
        )

    def defendercollision(self):
        return (
                self.distance(
                    self.strikerposition,
                    self.defenderposition
                ) < 0.40 #the defenders collision radius
        )
    def outofbounds(self):
        return (
                self.strikerposition[0] < 0
                or
                self.strikerposition[0] > self.pitchwidth
                or
                self.strikerposition[1] < 0
                or
                self.strikerposition[1] > self.pitchheight
        )

    ##RENDERING PYGAME ANIMATION FOR THE GAME
    #Creating a method to drawrobot first, then rendering
    def drawrobot(self, robotposition, robotangle, bodycolour, markercolour):
        robotx = int(robotposition[
                         0] * self.renderscalefactor)  # applying my scale factor to make values rendering friendly

        roboty = int(robotposition[
                         1] * self.renderscalefactor)  # applying my scale factor to make values rendering friendly
        robotlength = 50
        robotwidth = 34
        wheelwidth = 8
        wheellength = 28

        cosangle = np.cos(robotangle)
        sinangle = np.sin(robotangle)

        forwardvector = np.array([cosangle, sinangle])
        sidevector = np.array([-sinangle, cosangle])
        robotcenter = np.array([robotx, roboty])
        frontcenter = robotcenter + forwardvector * (robotlength / 2)
        backcenter = robotcenter - forwardvector * (robotlength / 2)
        frontleft = frontcenter + sidevector * (robotwidth / 2)
        frontright = frontcenter - sidevector * (robotwidth / 2)
        backleft = backcenter + sidevector * (robotwidth / 2)
        backright = backcenter - sidevector * (robotwidth / 2)
        robotpoints = [
            frontleft,
            frontright,
            backright,
            backleft
        ]
        robotpoints = [(int(point[0]), int(point[1])) for point in robotpoints]
        pygame.draw.polygon(  # body
            self.window,
            bodycolour,
            robotpoints
        )
        pygame.draw.polygon(  # outline
            self.window,
            (0, 0, 0),
            robotpoints,
            3
        )
        leftwheelcenter = robotcenter + sidevector * ((robotwidth / 2) + 6)
        rightwheelcenter = robotcenter - sidevector * ((robotwidth / 2) + 6)
        def drawwheel(wheelcenter):  # drawing robot wheels
            wheelfront = wheelcenter + forwardvector * (wheellength / 2)
            wheelback = wheelcenter - forwardvector * (wheellength / 2)
            wheelfrontleft = wheelfront + sidevector * (wheelwidth / 2)
            wheelfrontright = wheelfront - sidevector * (wheelwidth / 2)
            wheelbackleft = wheelback + sidevector * (wheelwidth / 2)
            wheelbackright = wheelback - sidevector * (wheelwidth / 2)
            wheelpoints = [
                wheelfrontleft,
                wheelfrontright,
                wheelbackright,
                wheelbackleft
            ]
            wheelpoints = [(int(point[0]), int(point[1])) for point in wheelpoints]
            pygame.draw.polygon(
                self.window,
                (40, 40, 40),
                wheelpoints
            )
        drawwheel(leftwheelcenter)
        drawwheel(rightwheelcenter)
        pygame.draw.circle(  #front striker marker
            self.window,
            markercolour,
            (int(frontcenter[0]), int(frontcenter[1])),
            5
        )
    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()

            self.window = pygame.display.set_mode(
                (
                    int(self.pitchwidth * self.renderscalefactor),
                    int(self.pitchheight * self.renderscalefactor)
                )
            )

            pygame.display.set_caption("Striker Environment")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.window.fill((30, 130, 50))

        pygame.draw.rect(  #drawing the pitch borders
            self.window,
            (255, 255, 255),
            pygame.Rect(
                0,
                0,
                int(self.pitchwidth * self.renderscalefactor),
                int(self.pitchheight * self.renderscalefactor)
            ),
            4
        )

        strikerx = int(self.strikerposition[
                           0] * self.renderscalefactor)  # applying my scale factor to make values rendering friendly
        strikery = int(self.strikerposition[
                           1] * self.renderscalefactor)  # applying my scale factor to make values rendering friendly

        ballx = int(
            self.ballposition[0] * self.renderscalefactor)  # applying my scale factor to make values rendering friendly
        bally = int(
            self.ballposition[1] * self.renderscalefactor)  # applying my scale factor to make values rendering friendly

        goalx = int(
            self.goalposition[0] * self.renderscalefactor)  # applying my scale factor to make values rendering friendly
        goaly = int(
            self.goalposition[1] * self.renderscalefactor)  # applying my scale factor to make values rendering friendly

        pygame.draw.rect(  # drawing goal
            self.window,
            (255, 255, 0),
            pygame.Rect(
                goalx - 10,
                goaly - 50,
                20,
                100
            )
        )
        pygame.draw.line( #center line on the pitch
            self.window,
            (255, 255, 255),
            (
                int((self.pitchwidth / 2) * self.renderscalefactor),
                0
            ),
            (
                int((self.pitchwidth / 2) * self.renderscalefactor),
                int(self.pitchheight * self.renderscalefactor)
            ),
            2
        )


        pygame.draw.circle( #center circle on the pitch
            self.window,
            (255, 255, 255),
            (
                int((self.pitchwidth / 2) * self.renderscalefactor),
                int((self.pitchheight / 2) * self.renderscalefactor)
            ),
            int(0.7 * self.renderscalefactor),
            2
        )

        pygame.draw.circle(  #drawing ball shadow and outline
            self.window,
            (0, 0, 0),
            (ballx, bally),
            12
        )

        pygame.draw.circle(  #drawing the ball
            self.window,
            (255, 255, 255),
            (ballx, bally),
            10
        )
        self.drawrobot(  #drawing defender robot
            self.defenderposition,
            self.defenderangle,
            (200, 0, 0),
            (255, 255, 255)
        )
        self.drawrobot( #drawing striker robot
            self.strikerposition,
            self.strikerangle,
            (0, 0, 255),
            (255, 0, 0)
            )
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None