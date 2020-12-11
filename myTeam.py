# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from util import manhattanDistance

#################
# Team creation #
#################

beliefs = []
initializeBeliefs = []

def createTeam(firstIndex, secondIndex, isRed,
               first = 'BustersPrioritiseTopEntryAgent', second = 'BustersPrioritiseBottomEntryAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class QLearningAgent(CaptureAgent):

    def __init__(self, index):
        self.weights = util.Counter()
        self.episodesSoFar = 0
        self.epsilon = 0.05
        self.gamma = 0.8
        self.alpha = 0.2
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.lastAction = None
        CaptureAgent.registerInitialState(self, gameState)


    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
          # Only half a grid position was covered
          return successor.generateSuccessor(self.index, action)
        else:
          return successor

    def chooseAction(self, state):

        """
        Picks among the actions with the highest Q(s,a).
        """
        legalActions = state.getLegalActions(self.index)
        bestAction = None

        if len(legalActions):
            if util.flipCoin(self.epsilon) == True:
                bestAction = random.choice(legalActions)
            else:
                bestAction = self.computeActionFromQValues(state)
        self.lastAction = bestAction

        foodLeft = len(self.getFood(state).asList())

        if foodLeft <= 2:
          bestDist = 9999
          for a in legalActions:
            successor = self.getSuccessor(state, a)
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(self.start, pos2)
            if dist < bestDist:
              bestAction = a
              bestDist = dist
        return bestAction

        if gameState.getAgentState(self.index).numCarrying == min(foodLeft, 5) :
            bestDist = 9999

            for action in legalActions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['score'] = self.getScore(successor)
        if not self.red:
          features['score'] *= -1
        return features

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = state.getLegalActions(self.index)
        bestActions = 0
        bestValue = -999999

        for action in actions:
            value = self.getQValue(state, action)
            if value > bestValue:
                bestActions = [action]
                bestValue = value
            elif value == bestValue:
                bestActions.append(action)
        if bestActions == 0:
            return Directions.STOP

        return random.choice(bestActions)

    def getWeights(self):
        return self.weights

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        qValueArray = []
        actions = state.getLegalActions(self.index)

        for action in actions:
            values = self.getQValue(state, action)
            qValueArray.append(values)

        if len(actions) == 0:
            return 0
        else:
            return max(qValueArray)

    def getQValue(self, state, action):
        features = self.getFeatures(state, action)
        weights = self.getWeights()
        return features * weights

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        qValue = self.getQValue(state, action)
        sample = (reward + self.gamma * self.computeValueFromQValues(nextState))

        self.weights[(state, action)] = ((1 - self.alpha) * qValue) + (self.alpha * sample)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)

class ApproximateQAgent(QLearningAgent):

    def registerInitialState(self, gameState):
        QLearningAgent.registerInitialState(self, gameState)

        self.lastDeposit = 0.0
        self.timeToDefend = 0.0
        self.getLegalPositions(gameState)
        self.favouriteOption = 0.0

    def __init__(self, index):
        QLearningAgent.__init__(self, index)
        self.weights = util.Counter()
        self.weights = {'successorScore': 100, 'foodDistance': -1, 'stop': -1000, 'legalActions': 100, 'capsuleValue': 100, 'goBackHome': -1,'ghostDistance': 5, 'killEnemy': -100}
        self.distFromPowerCapsule = 3
        self.enemyDistance = 5
        self.attackerDistance = 5
        self.mapOfAvailableActions = {}
        self.initializedPosition = False

    def getScoreDifference(self, gameState):
        if self.red:
          return gameState.getScore()
        else:
          return -1 * gameState.getScore()

    def getLegalPositions(self, gameState):
        if not self.initializedPosition:
          walls = gameState.getWalls()
          self.legalPositions = []
          for w in range(walls.width):
            for h in range(walls.height):
              if not walls[w][h]:
                self.legalPositions.append((w, h))
          self.initializedPosition = True
        return self.legalPositions

    def shouldRunHome(self, gameState):
        scoreDifference = self.getScoreDifference(gameState)
        numCarrying = gameState.getAgentState(self.index).numCarrying
        return (gameState.data.timeleft < 100
          and scoreDifference <= 0
          and numCarrying > 0
          and numCarrying >= abs(scoreDifference))

    def getFeatures(self, gameState, action):
        self.observeStates(gameState)
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)

        if len(foodList) > 0:
          bustersDistance = min([self.getBustersDistance(myPos, food) for food in foodList])
          features['foodDistance'] = bustersDistance

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]
        enemyGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
        scaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]

        enemyGhostsDist = []
        for i in self.getOpponents(successor):
          enemy = successor.getAgentState(i)
          if enemy in enemyGhosts:
            if True:
              enemyGhostsDist.append(self.getMazeDistance(myPos, self.getPossibleGhostPosition(i)))
            else:
              enemyGhostsDist.append(self.getMazeDistance(myPos, enemy.getPosition()))
        if len(enemyGhostsDist) > 0:
          smallestDist = min(enemyGhostsDist)
          features['ghostDistance'] = smallestDist

        features['capsuleValue'] = self.getcapsuleValue(myPos, successor, scaredGhosts)
        features['killEnemy'] = self.getKillEnemyWeight(myPos, enemyPacmen)

        if myState.numReturned != self.lastDeposit:
          self.timeToDefend = 100.0
          self.lastDeposit = myState.numReturned
        if self.timeToDefend > 0:
          self.timeToDefend -= 1
          features['killEnemy'] *= 100
        if len(self.getFoodYouAreDefending(successor).asList()) <= 2:
          features['killEnemy'] *= 100
        if action == Directions.STOP:
            features['stop'] = 1

        features['legalActions'] = self.getLegalActionModifier(gameState, 1)
        features['goBackHome'] = self.getDepositValue(myPos, gameState, myState)
        features['goBackHome'] += self.getHomeDistance(myPos, features['ghostDistance'])

        if self.shouldRunHome(gameState):
          features['goBackHome'] = self.getMazeDistance(self.start, myPos) * 10000

        return features

    def getWeights(self):
        return self.weights

    def getBustersDistance(self, myPos, food):
        return self.getMazeDistance(myPos, food) + abs(self.favouriteOption - food[1])

    def getcapsuleValue(self, myPos, successor, scaredGhosts):
        powerCapsules = self.getCapsules(successor)
        minDistance = 0
        if len(powerCapsules) > 0 and len(scaredGhosts) == 0:
          distances = [self.getMazeDistance(myPos, pellet) for pellet in powerCapsules]
          minDistance = min(distances)
        return max(3 - minDistance, 0)

    def getDepositValue(self, myPos, gameState, myState):
        if myState.numCarrying >= 8:
          return self.getMazeDistance(self.start, myPos)
        else:
          return 0

    def getHomeDistance(self, myPos, smallestGhostPosition):
        if smallestGhostPosition > self.attackerDistance or smallestGhostPosition == 0:
          return 0
        else:
          return self.getMazeDistance(self.start, myPos) * 1000

    def getKillEnemyWeight(self, myPos, enemyPacmen):
        if len(enemyPacmen) > 0:
          dists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemyPacmen]
          if len(dists) > 0:
            smallestDist = min(dists)
            return smallestDist
        return 0

    def getPossibleGhostPosition(self, ghostAgentIndex):
        return max(beliefs[ghostAgentIndex])

    def getLegalActionModifier(self, gameState, numLoops):
        legalActions = gameState.getLegalActions(self.index)
        numActions = len(legalActions)
        for legalAction in legalActions:
          if numLoops > 0:
            newState = self.getSuccessor(gameState, legalAction)
            numActions += self.getLegalActionModifier(newState, numLoops - 1)
        return numActions

    def initializeBelief(self, opponentLocation, gameState):
        self.belief = util.Counter()
        for p in self.getLegalPositions(gameState):
            self.belief[p] = 1.0
        self.belief.normalize()
        beliefs[opponentLocation] = self.belief

    def initializeBeliefs(self, gameState):
        beliefs.extend([None for x in range(len(self.getOpponents(gameState)) + len(self.getTeam(gameState)))])
        for opponent in self.getOpponents(gameState):
          newPosDist = self.initializeBelief(opponent, gameState)
        initializeBeliefs.append(newPosDist)

    def observeStates(self, gameState):
        if len(initializeBeliefs):
            for opponent in self.getOpponents(gameState):
                self.observeState(gameState, opponent)
        else:
          self.initializeBeliefs(gameState)

    def observeState(self, gameState, opponentLocation):
        noisyDistance = gameState.getAgentDistances()[opponentLocation]
        pacmanPosition = gameState.getAgentPosition(self.index)
        possiblePosition = gameState.getAgentPosition(opponentLocation)

        allPossible = util.Counter()
        for p in self.getLegalPositions(gameState):
            if possiblePosition != None:
                allPossible[possiblePosition] = 1.0
                beliefs[opponentLocation] = allPossible
            else:
                trueDistance = util.manhattanDistance(p, pacmanPosition)
                emissionModel = gameState.getDistanceProb(trueDistance, noisyDistance)
                if emissionModel > 0:
                    allPossible[p] = emissionModel * (beliefs[opponentLocation][p] + 0.0001)

        allPossible.normalize()
        beliefs[opponentLocation] = allPossible

class ReflexCaptureAgent(CaptureAgent):

    """
    A base class  for reflex agents that chooses score-maximizing actions

    """

    def registerInitialState(self, gameState):

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)


    def chooseAction(self, gameState):

        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        ## if theres only 2 or less food left on grid it will start going home to be safe
        if foodLeft <= 2:
            bestDist = 9999
            for a in actions:
                successor = self.getSuccessor(gameState, a)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.distancer.getDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = a
                    bestDist = dist
            return bestAction

        if gameState.getAgentState(self.index).numCarrying == min(foodLeft, 5) :
            bestDist = 9999

            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.distancer.getDistance(self.start,pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    # Method inherited from baselineTeam.py
    def getSuccessor(self, gameState, action):

        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # Method inherited from baselineTeam.py
    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    # Method inherited from baselineTeam.py
    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        if not self.red:
            features['successorScore'] *= -1
        return features

    # Method inheritedd from baselineTeam.py
    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

class DefensiveHueristicReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def registerInitialState(self, gameState):
    ReflexCaptureAgent.registerInitialState(self, gameState)
    ## to setup the best entry points
    self.homePoints = []
    if self.red:
        central = (gameState.data.layout.width - 2) / 2
    else:
        central = ((gameState.data.layout.width - 2) / 2) + 1

    for height in range(1, gameState.data.layout.height - 1):
        if not gameState.hasWall(int(central), height):
            self.homePoints.append((central, height))

    self.entryPoints = self.homePoints

    if not self.red:
        x = 1
        y = 1
    else:
        x = gameState.data.layout.width - 2
        y = gameState.data.layout.height - 2

    self.ghostPos = (x, y)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    defendingFoodList = self.getFoodYouAreDefending(gameState).asList()
    if self.red:
        self.hoverPoint = min(defendingFoodList)
    else:
        self.hoverPoint = max(defendingFoodList)

    if not self.red:
        x = 1
        y = 1
    else:
        x = gameState.data.layout.width - 2
        y = gameState.data.layout.height - 2

    self.hoverPoint = (x, y)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.distancer.getDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
      features['onDefence'] = 1

    # Incentive to hang around hover position
    if len(invaders) == 0:
        features['hoverPlace'] = (self.distancer.getDistance(myPos, self.hoverPoint))

    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)
    capsules = self.getCapsules(successor)
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    threats = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    for food in foodList:
        foodArray = []
        foodDistance = self.distancer.getDistance(myPos, food)
        foodArray.append(foodDistance)
        if self.red:
            minFoodDistance = min(foodArray)
        else:
            minFoodDistance = max(foodArray)
        features['distanceToFood'] = minFoodDistance

    for capsule in capsules:
        capsuleArray = []
        capDistance = self.distancer.getDistance(myPos, capsule)
        capsuleArray.append(capDistance)
        minCapDistance = min(capsuleArray)
        features['capsuleDistance'] = minCapDistance

    ## to run away from ghosts
    features['enemyDistance'] = 999999
    if len(threats) > 0:
        minDistanceToThreat = min(
            [self.distancer.getDistance(successor.getAgentState(self.index).getPosition(), threat.getPosition()) for
             threat in threats])
        nearestThreat = [threat for threat in threats if
                         self.distancer.getDistance(successor.getAgentState(self.index).getPosition(),
                                                    threat.getPosition()) == minDistanceToThreat]
        if nearestThreat[0].scaredTimer > 0:
            features['enemyDistance'] = 999999
        elif successor.getAgentState(self.index).isPacman:
            features['enemyDistance'] = minDistanceToThreat

    if action == Directions.STOP: features['stop'] = 1
    reverseDirection = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == reverseDirection: features['reverse'] = 1

    ## to get the best entry in the opponent half
    if not gameState.getAgentState(self.index).isPacman:
        bestEntry = (0, 0)
        d_ghostToEntry = 0.0
        for ep in self.entryPoints:
            d = self.distancer.getDistance(self.ghostPos, ep)
            if d > d_ghostToEntry:
                d_ghostToEntry = d
                bestEntry = ep
        features['distanceToEntry'] = self.distancer.getDistance(myPos, bestEntry)

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -100, 'onDefense': 1, 'invaderDistance': -15, 'reverse': -20, 'hoverPlace': -15,
            'successorScore': 100, 'distanceToFood': -4, 'enemyDistance': 15, 'distanceToEntry': -10, 'capsuleDistance': -4, 'stop': -300}

class OffensiveReflexAgent(ReflexCaptureAgent):

    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)
        ## to setup the best entry points
        self.homePoints = []
        if self.red:
            central = (gameState.data.layout.width - 2) / 2
        else:
            central = ((gameState.data.layout.width -2) / 2) + 1

        for height in range(1, gameState.data.layout.height-1):
            if not gameState.hasWall(int(central), height):
                self.homePoints.append((central, height))

        self.entryPoints = self.homePoints

        if not self.red:
            x = 1
            y = 1
        else:
            x = gameState.data.layout.width - 2
            y = gameState.data.layout.height - 2

        self.ghostPos = (x,y)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successScore'] = -len(foodList)
        capsules = self.getCapsules(successor)
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        threats = [a for a in enemies if a.getPosition() != None]

        for food in foodList:
            foodArray = []
            foodDistance = self.distancer.getDistance(myPos, food)
            foodArray.append(foodDistance)
            if self.red: minFoodDistance = min(foodArray)
            else: minFoodDistance = max(foodArray)
            features['distanceToFood'] = minFoodDistance

        for capsule in capsules:
            capsuleArray = []
            capDistance = self.distancer.getDistance(myPos, capsule)
            capsuleArray.append(capDistance)
            minCapDistance = min(capsuleArray)
            features['capsuleDistance'] = minCapDistance


        ## to run away from ghosts
        features['enemyDistance'] = 999999
        if len(threats) > 0:
            minDistanceToThreat = min([self.distancer.getDistance(successor.getAgentState(self.index).getPosition(), threat.getPosition()) for threat in threats])
            nearestThreat = [threat for threat in threats if
                             self.distancer.getDistance(successor.getAgentState(self.index).getPosition(), threat.getPosition()) == minDistanceToThreat]
            if nearestThreat[0].scaredTimer > 0:
                features['enemyDistance'] = 999999
            elif successor.getAgentState(self.index).isPacman:
                features['enemyDistance'] = minDistanceToThreat



        if action == Directions.STOP: features['stop'] = 1
        reverseDirection = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == reverseDirection: features['reverse']  = 1

        ## to get the best entry in the opponent half
        if not gameState.getAgentState(self.index).isPacman:
            bestEntry = (0,0)
            d_ghostToEntry = 0.0
            for ep in self.entryPoints:
                d = self.distancer.getDistance(self.ghostPos,ep)
                if d > d_ghostToEntry:
                    d_ghostToEntry = d
                    bestEntry = ep
            features['distanceToEntry'] = self.distancer.getDistance(myPos,bestEntry)

        return features

    def getWeights(self, gameState, action):
        return {'successScore': 100, 'distanceToFood': -4, 'enemyDistance': 10, 'distanceToEntry': -10, 'capsuleDistance': -4, 'reverse': -20, 'stop': -300}

class BustersPrioritiseTopEntryAgent(ApproximateQAgent):

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.favouriteOption = gameState.data.layout.height

class BustersPrioritiseBottomEntryAgent(ApproximateQAgent):

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.favouriteOption = 0.0