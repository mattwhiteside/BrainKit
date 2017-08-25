//
//  BiLSTM.swift
//  BrainKit
//
//  Created by Matthew Whiteside on 2/4/17.
//  Copyright © 2017 mattwhiteside. All rights reserved.
//

import Foundation
import MultiLinAlg

extension ExpressibleByFloatLiteral{
   public static var e : Self{
      get {
         return 2.71828182846
      }
   }
}




internal protocol Nameable{
   var name:String{
      get
   }
}


internal protocol Layer : Nameable{
   var outputActivations:Matrix{
      get
   }
   
   
}


internal protocol SourceLayer:class, Layer{
   func feedForward(_ inputs:Matrix) -> Void
}


internal protocol NonSourceLayer:class, Layer{
   var inputPorts:OrderedDictionary<String,InputPort>{
      get
      set
   }
   
   var outputErrors:Matrix{
      get
   }

   func feedForward() -> Void
   func feedBack() -> Void

   func addIncomingConnection(from layer: Layer,
                              to inputPortName: String,
                              dimensions:[Int],
                              offset:Int) -> Void
}

internal protocol NetworkOutput: NonSourceLayer{
//   var errorMap:Dictionary<String, Rational>{
//      get
//      set
//   }
//   var errors:Dictionary<String, Rational>{
//      get
//      set
//   }
//   var normFactors: Dictionary<String, Rational>{
//      get
//      set
//   }

   func calculateErrors(for inputs:Matrix) -> (error:Rational,
                                               errorMap:Dictionary<String, Rational>)
}

class InputPort{
   var biases: Vector
   var biasOptimizerDeltas: Vector
   var connections = [Connection]()
   let activationFunc: (Rational) -> Rational
   var peepholeWeights:Vector?
   var peepholeOptimizerDeltas:Vector?
   var errors:Vector?
   var derivatives:Vector?

   init(activationFunc:@escaping (Rational) -> Rational, numCells: Int){
      self.activationFunc = activationFunc
      self.biases = Vector(repeating: 0.0, count: numCells)
      self.biasOptimizerDeltas = Vector(repeating: 0.0, count: numCells)
   }
}



class Connection{
   let from:Layer
   var weights:Matrix
   var weightOptimizerDeltas:Matrix
   let offset:Int
   var errors:Matrix?
   var derivatives:Matrix?
   
   
   init(from:Layer, rows: Int, cols: Int, offset: Int = 0){
      self.from = from
      self.offset = offset
      let zeroMatrix = Matrix.zero(rows: rows, cols: cols)
      weights = zeroMatrix
      weightOptimizerDeltas = zeroMatrix
   }
}



extension NonSourceLayer{
   func addIncomingConnection(from layer: Layer,
                              to targetPort: String,
                              dimensions: [Int],
                              offset: Int = 0) -> Void {
      
      inputPorts[targetPort]!.connections.append(Connection(from: layer,
                                                            rows: dimensions[0],
                                                            cols: dimensions[1],
                                                            offset: offset))
      
   }
}


internal class InputLayer:SourceLayer{
   
   static let _name = "input layer"
   var _outputActivations = Matrix()
   var outputActivations: Matrix{
      get{
         return _outputActivations
      }
   }
   var name:String{
      get{
         return InputLayer._name
      }
   }

   func feedForward(_ inputs:Matrix) -> Void{
      _outputActivations = inputs
   }
   
}


internal class SoftmaxLayer : NetworkOutput{
   

   var inputPorts = OrderedDictionary<String, InputPort>()
   let inputPortName:String
   var inputPort:InputPort
   let numTargets:Int
   let _name:String
   var _outputActivations:Matrix!
   var errors = Dictionary<String, Rational>()
   var errorMap = Dictionary<String, Rational>()
   var normFactors = Dictionary<String, Rational>()
   let labels:[String]
   var outputErrors = Matrix()
   var inputErrors = Matrix()
   var inputActivations:Matrix?
   
   
   var outputActivations: Matrix{
      get{
         return _outputActivations
      }
   }

   var name:String{
      get{
         return _name
      }
   }
   
   
   
   init(name: String, numTargets:Int, inputPortName:String, labels:[String]){
      _name = name
      self.numTargets = numTargets
      self.inputPortName = inputPortName
      let stub:(Rational) -> Rational = {(x:Rational) in fatalError("output layer not implemented yet")}
      self.inputPorts[inputPortName] = InputPort(activationFunc: stub, numCells: numTargets)
      self.inputPort = self.inputPorts[inputPortName]!
      self.labels = labels
      //self.outputErrors = Vector(repeating: 0.0, count: numTargets)
   }
   
   
   func feedForward() -> Void{
      

      
      for connection in inputPort.connections{
         let incoming = (connection.from as! LSTMLayer)
         incoming.feedForward()
         if inputActivations == nil{
            inputActivations = Matrix(repeating: inputPort.biases, count: incoming.outputActivations.count)
         }
         let outputActivations = incoming.outputActivations
         inputActivations = inputActivations! + (outputActivations ⋅ connection.weights)
      }
      
      
      var __outputActivations = Matrix.zero(rows: inputActivations!.count, cols: numTargets)
      //var logActivations = __outputActivations
      for (t,input) in inputActivations!.enumerated(){
//         let min = input.min()
//         let max = input.max()
//         let offset = (min! + max!)/2.0
//         let unnormedLogActivations = input.map{(x:Double) -> (_exp:Double,_log:Double) in
//            let shiftedX = x - offset
//            let result:Rational
//            if shiftedX <= expMin{
//               result = logInfinity
//            } else {
//               result = log(shiftedX)
//            }
//            
//            return (shiftedX, result)
//         }
//         let unnormedActivations = unnormedLogActivations.map{safe_exp($0.0)}
//         
//         let denom = unnormedActivations.sum()
//         __outputActivations[t] = unnormedActivations.map{
//            $0 / denom
//         }
         __outputActivations[t] = softmax(input)
//         logActivations[t] = unnormedLogActivations.map{
//            $0._log / log(denom)
//         }
      }
      
      _outputActivations = __outputActivations
      let rows = outputActivations.count
      let cols = outputActivations[0].count
      outputErrors = Matrix.zero(rows: rows, cols: cols)
      self.inputErrors = Matrix.zero(rows: rows, cols: cols)

   }
   
   func setError(index:Int, targetClass:Int) -> Rational{
      let realMin = Rational.leastNonzeroMagnitude
      let targetProb = max(realMin, outputActivations[index][targetClass])
      outputErrors[index][targetClass] = -1.0/targetProb
      return log(targetProb)
   }
   
   func calculateErrors(for inputs: Matrix) -> (error: Rational, errorMap:Dictionary<String, Rational>){
      
      var confusionMatrix = Matrix.zero(rows: numTargets, cols: numTargets)
      var crossEntropyError:Rational = 0;
      var outputs = [Int]()
      var targets = Vector(repeating: 0, count: numTargets)
//      targets.reshape(seq.targetClasses.seq_shape(), 0);
//      real_t crossEntropyError = 0;
      let seqSize = 1//TODO: derive this from the shape of the input sequence
      for i in 0..<seqSize{
         var indexOfMaxVal = 0
         for (j,activation) in outputActivations[i].enumerated(){
            if activation > outputActivations[i][indexOfMaxVal]{
               indexOfMaxVal = j
            }
         }
         let outputClass = indexOfMaxVal
         print("output class = \(outputClass)")
         outputs.append(outputClass)
         let targetClass = 0//TODO: derive this/figure out it's original purpose
         if targetClass >= 0 {
            targets[i] = 1
            crossEntropyError -= setError(index: i, targetClass: targetClass);
            confusionMatrix[targetClass][outputClass] += 1
         }
      }

      var numTargetsByClass = Vector(repeating: 0.0, count: numTargets)
      var numErrorsByClass = numTargetsByClass
                                                   
      //operations related to the diagonal of the confusion matrix
      for (i, row) in confusionMatrix.enumerated(){
         let sum = row.sum()
         numTargetsByClass[i] = sum
         numErrorsByClass[i] = sum - row[i]
      }
      var errorMap = [String: Rational]()
      var normFactors = [String: Rational]()
                                                   
      var _numTargets = numTargetsByClass.sum()
      if _numTargets > 0{
         errorMap["crossEntropyError"] = crossEntropyError;
         errorMap["classificationError"] = numErrorsByClass.sum() / Rational(numTargets);
         for i in 0..<confusionMatrix.count{
            if (numTargetsByClass[i] > 0){
               
               errorMap["_" + labels[i]] = Rational(numErrorsByClass[i]) / Rational(numTargets);
               if(confusionMatrix.count > 2)
               {
                  let v = confusionMatrix[i]
                  
                  for j in 0..<v.count{
                     if ((j != i) && v[j] > 0){
                        errorMap["_" + labels[i] + "->" + labels[j]] = v[j] / Rational(numTargets);
                     }
                  }
               }
            }
         }
      }
      return (crossEntropyError, errorMap)
   }
   
   func feedBack(){
      print("feeding back output layer")
      for i in (0..<outputActivations.count).reversed(){
         let Z = outputActivations[i] ⋅ outputErrors[i];
         for j in 0..<outputActivations[i].count{
            self.inputErrors[i][j] = outputActivations[i][j] * (outputErrors[i][j] - Z)
         }
      }
      for (_,port) in self.inputPorts{
         port.errors = self.inputErrors ⋅ port.biases
         
         for connection in port.connections{
            connection.errors = self.inputErrors ⋅ connection.weights.transposed()
         }
      }
      for (_,port) in self.inputPorts{
         //in the original RNNLib, this call is an empty abstract base-class method
         
         print("TODO: check why we're using index zero here")
         //let slice = Matrix(self.inputErrors.transposed()[range]).transposed()
         port.derivatives = self.inputErrors.collapse()
         
         for connection in port.connections {
            var accumulator = Matrix.zero(rows: self.inputErrors.cols,
                                          cols: connection.from.outputActivations.cols)
            //TODO: figure out why this needs to be reversed
            let opActsReversed = Matrix(connection.from.outputActivations)
            for (offset, ipErrorVec) in self.inputErrors.enumerated(){
               //TODO: why do the
               let opActivationVec = opActsReversed[offset]
               let outerProd = (ipErrorVec ⊗ opActivationVec)
               accumulator = accumulator + outerProd
               print("ok")
            }
            connection.derivatives = accumulator
         }
      }
      print("done")

   }
}



public class BiLSTM{
   let hiddenLayers:[LSTMLayer]
   let outputLayer:SoftmaxLayer
   let inputDim:Int
   let inputLayer = InputLayer()
   let targetLabels: Array<String>
   var errors = Dictionary<String, Rational>()
   var normFactors = Dictionary<String, Rational>()
   
   
   public init?(config: Dictionary<String, Any>) {
      if let hiddenSize = config["hiddenSize"] as? Int,
         let _inputDim = config["inputSize"] as? Int,
         let targetLabels = config["targetLabels"] as? Array<String>,
         let weights = config["weightContainer"] as? Dictionary<String,Dictionary<String,Any>>,
         let __biases = weights["biases"] as? Dictionary<String, Dictionary<String, Array<Double>>>,
         let _inputWeights = weights["inputs"] as? Dictionary<String, Dictionary<String, Array<Double>>>,
         let hiddenWeights = weights["hidden"] as? Dictionary<String, Dictionary<String, Dictionary<String, Array<Double>>>>,
         let outputBiases = __biases["to_output"]
         
      {
         var _hiddenLayers = [LSTMLayer]()
         self.inputDim = _inputDim
         self.targetLabels = targetLabels
         self.outputLayer = SoftmaxLayer(name: "softmax output layer",
                                         numTargets: targetLabels.count,
                                         inputPortName: "inputs",
                                         labels: targetLabels)
         let numCells = hiddenSize

         for l in [0,1]{
            let layerName = "hidden layer \(l)"
            let direction = l == 0 ? LSTMLayer.Direction.backward : LSTMLayer.Direction.forward
            let hiddenLayer = LSTMLayer(name: layerName,
                                        inputDim: inputDim,
                                        useMetal: false,
                                        numCells: hiddenSize,
                                        direction: direction)

            
            for (portName, _) in hiddenLayer.inputPorts{
               hiddenLayer.addIncomingConnection(from: inputLayer,
                                                 to: portName,
                                                 dimensions: [numCells, inputDim])
               
               hiddenLayer.addIncomingConnection(from: hiddenLayer,
                                                 to: portName,
                                                 dimensions: [numCells, numCells],
                                                 offset: -1)
            }
            
            outputLayer.addIncomingConnection(from: hiddenLayer,
                                              to: outputLayer.inputPortName,
                                              dimensions:[targetLabels.count, numCells])
            
            let inputWeights = (_inputWeights["to_hidden_0_\(l)"]?["weights"])!
            //optimizer deltas are not used anywhere yet, but the
            //idea is that they will be
            let inputOptimizers = (_inputWeights["to_hidden_0_\(l)"]?["weight_optimiser_deltas"])!
            
            let recurrentWeights = (hiddenWeights["0_\(l)"]?["to_hidden_0_\(l)"]?["weights"])!
            let recurrentOptimizers = (_inputWeights["to_hidden_0_\(l)"]?["weight_optimiser_deltas"])!

            let biases = (__biases["to_hidden_0_\(l)"]?["weights"])!
            let biasOptimizerDeltas = (__biases["to_hidden_0_\(l)"]?["weight_optimiser_deltas"])!
            
            let peepholeWeights = (hiddenWeights["0_\(l)"]?["peepholes"]?["weights"])!
            let peepholeOptimizers = (hiddenWeights["0_\(l)"]?["peepholes"]?["weight_optimiser_deltas"])!
            
            let outputWeights = (hiddenWeights["0_\(l)"]?["to_output"]?["weights"])!
            let outputOptimizers = (hiddenWeights["0_\(l)"]?["to_output"]?["weight_optimiser_deltas"])!
            let portCount = hiddenLayer.inputPorts.count
            for i in 0..<numCells{
               for j in 0..<inputDim{
                  var k = 0
                  for (_, port) in hiddenLayer.inputPorts{
                     let val = inputWeights[inputDim*(portCount*i+k) + j]
                     port.connections[0].weights[i,j] = val
                     k += 1
                  }
               }
               for j in 0..<numCells{
                  var k = 0
                  for (_, port) in hiddenLayer.inputPorts{
                     let val = recurrentWeights[numCells*(portCount*i+k) + j]
                     port.connections[1].weights[i,j] = val
                     k += 1
                  }
               }
               var k = 0
               for (_, port) in hiddenLayer.inputPorts{
                  let index = portCount*i+k
                  let val = biases[index]
                  port.biases[i] = val
                  k += 1
               }
               
               
               let portsWithPeepholesCount = 3
               k = 0
               for (portName, port) in hiddenLayer.inputPorts {
                  guard portName != "blockInputs" else{
                     continue
                  }
                  if let _ = port.peepholeWeights {
                     //TODO: see if I can accomplish this flow 
                     //control with some sort of guard statement
                  } else {
                     port.peepholeWeights = Vector(repeating: 0.0, count: numCells)
                     port.peepholeOptimizerDeltas = Vector(repeating: 0.0, count: numCells)
                  }
                  let index = portsWithPeepholesCount*i + k
                  let val = peepholeWeights[index]
                  port.peepholeWeights![i] = val
                  k += 1
               }

            


            }
            
            for i in 0..<targetLabels.count{
               for j in 0..<numCells{
                  let index = numCells*i + j
                  let weight = outputWeights[index]
                  outputLayer.inputPort.connections[l].weights[i,j] = weight
               }
            }
            
            outputLayer.inputPort.connections[l].weights = outputLayer.inputPort.connections[l].weights.transposed()
            
            
            if l == 0{
               outputLayer.inputPort.biases = outputBiases["weights"]!
            }

            _hiddenLayers.append(hiddenLayer)


         }

         hiddenLayers = _hiddenLayers
      } else {
         return nil
      }
   }
   
   
   func feedForward(_ inputs:Array<Vector>) -> Void {
      inputLayer.feedForward(inputs)
      outputLayer.feedForward()
   }

   
   func calculateOutputErrors(for inputs:Matrix) -> Rational {
      var error:Rational = 0
      errors.removeAll()
      (error, self.errors) = outputLayer.calculateErrors(for: inputs)
      return error
   }
   
   func calculateLoss(for inputs:Matrix) -> Rational{
      feedForward(inputs)
      return calculateOutputErrors(for: inputs)
   }
   
   func calculateLossAlt(outputActivations:Matrix, expected: [[Int]], numExamples:Int) -> Rational{
      var L = Vector.zero(dim: expected.count)
      var confusionMatrix = Matrix.id(rows: outputActivations.count)
      
      
      for i in 0..<expected.count {
         let v = confusionMatrix[i] ⊙ log(outputActivations[i])
         L[i] = v.sum()
      }
      
      return L.sum() / Rational(numExamples)
   }

   
   func feedBack(){
      outputLayer.feedBack()
      for layer in hiddenLayers {
         layer.feedBack()
      }
   }
   
   func train(with inputs:Matrix) -> Rational{
      let error = calculateLoss(for: inputs)
      feedBack()
      return error
   }
   
   public func classify(_ featureDataSeq:Matrix) throws -> [(Int, Rational)]{
      let error = train(with: featureDataSeq)
      //todo implement the correct return value
      return [(0,error)]
   }

}
