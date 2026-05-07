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
        self.maxlinearvelocity = 3.0
        self.maxangularvelocity = 3.0
        #Wheel actions = [left wheel torque, right wheel torque]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(         #my base of observations[distance_to_ball, angle_to_ball,distance_to_goal, angle_to_goal, striker_velocity, striker_angular_velocity]
            low=-np.inf,
            high=np.inf,
            shape=(6,),
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
        self.strikerposition = np.array([1.0, 3.0], dtype=np.float32)
        self.strikerangle = 0.0
        self.strikervelocity = 0.0
        self.strikerangularvelocity = 0.0

        self.ballposition = np.array([5.0, 3.0], dtype=np.float32) #ball position at the starting state
        self.goalposition = np.array([9.5, 3.0], dtype=np.float32) #goal position at starting state

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

        self.strikerposition += direction * forwardspeed * self.dt           #this updates the strikers direction
        if self.distance(self.strikerposition, self.ballposition) < 0.35:    #for if ball collides
            self.ballposition += direction * 0.25

        reward = self.calculatereward()
        terminated = False
        truncated = False

        #POINT ALLOCATIONS FOR RL
        if self.goalscored():
            reward += 100
            terminated = True
        if self.outofbounds():
            reward -= 10
            terminated = True
        # Episode timeout
        if self.steps >= self.maxsteps:
            truncated = True

        sensorvector = self.getsensorvalues()
        info = {}
        return sensorvector, reward, terminated, truncated, info

    def getsensorvalues(self):

        distance_to_ball, angle_to_ball = self.relativeinfo(
            self.ballposition
        )
        distance_to_goal, angle_to_goal = self.relativeinfo(
            self.goalposition
        )
        sensorvector = np.array([
            distance_to_ball,
            angle_to_ball,
            distance_to_goal,
            angle_to_goal,
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

    def goalscored(self):
        return (
                self.ballposition[0] >= 9.3
                and
                2.4 <= self.ballposition[1] <= 3.6
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

#DRAWING STRIKER
        robotlength = 50
        robotwidth = 34
        wheelwidth = 8
        wheellength = 28

        cosangle = np.cos(self.strikerangle)
        sinangle = np.sin(self.strikerangle)
        forwardvector = np.array([cosangle, sinangle])
        sidevector = np.array([-sinangle, cosangle])

        robotcenter = np.array([strikerx, strikery])
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
        pygame.draw.polygon(  #body
            self.window,
            (0, 0, 255),
            robotpoints
        )

        pygame.draw.polygon(  #outline
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
            (255, 0, 0),
            (int(frontcenter[0]), int(frontcenter[1])),
            5
        )
        directionx = strikerx + int(np.cos(self.strikerangle) * 25) #striker direction lines drawing
        directiony = strikery + int(np.sin(self.strikerangle) * 25)
        pygame.draw.line(
            self.window,
            (255, 0, 0),
            (strikerx, strikery),
            (directionx, directiony),
            3
        )
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None