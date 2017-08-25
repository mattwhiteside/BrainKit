//
//  LSTMLayer.swift
//  BrainKit
//
//  Created by Matthew Whiteside on 1/29/17.
//  Copyright © 2017 mattwhiteside. All rights reserved.
//

import Darwin
import Accelerate
import MultiLinAlg


//extension Array where Iterator.Element:Collection & ExpressibleByFloatLiteral & FloatingPoint{
//   init(rows: Int, cols: Int, values: [Iterator.Element]){
//      let zeros = Array<Iterator.Element>(repeating: 0.0, count: cols)
//      var _self = [[Iterator.Element]](repeating: zeros, count: rows)
//      for row in 0..<rows{
//         for col in 0..<cols{
//            _self[row][col] = values[row*cols + col]
//         }
//      }
//   }
//}

public enum NNetParamKey {
   case inputs
   case inputGates
   case outputGates
   case forgetGates
}


class LSTMLayer : NonSourceLayer{
   var outputErrors:Matrix

   var inputPorts = OrderedDictionary<String, InputPort>()
   var outputActivations = Matrix()
   
   let name:String
   enum Direction{
      case forward
      case backward
   }
   let direction:Direction
   
   
   //static let peepholeConnections = inputPorts.keys.filter{$0 != "blockInputs"}
   let inputDim: Int
   let numCells: Int
   //var GPU: MetalDevice!
   var useMetal: Bool
   
   
   
   public init(name: String,
               inputDim: Int,
               useMetal: Bool,
               numCells: Int,
               direction: Direction) {
      self.name = name
      self.inputDim = inputDim
      self.numCells = numCells
      self.useMetal = useMetal
      self.direction = direction
      
      self.inputPorts["inputGates"] = InputPort(activationFunc: σ, numCells: numCells)
      self.inputPorts["forgetGates"] = InputPort(activationFunc: σ, numCells: numCells)
      self.inputPorts["blockInputs"] = InputPort(activationFunc: BrainKit.tanh, numCells: numCells)
      self.inputPorts["outputGates"] = InputPort(activationFunc: σ, numCells: numCells)
      
      self.outputErrors = Matrix()
   }
   
   internal func feedForward() -> Void {
      let outputGate = inputPorts["outputGates"]!
      //assumes all incoming connections' sources have the same
      // number of output activations
      let inputCount = outputGate.connections.first!.from.outputActivations.count
      let zeros = Matrix.zero(rows: inputCount, cols: numCells)
      var states = zeros
      
      var activations = Dictionary<String, Matrix>()
      
      
      outputActivations.removeAll()
      
      for (portName, port) in inputPorts{
         activations[portName] = zeros
         activations[portName]![0] = port.biases
         for connection in port.connections{
            if connection.from.name == self.name{
               //no recurrent connection to the first cell in the layers
               continue
            } else {
               let rhs:Vector
               if direction == .backward && (connection.from is SourceLayer) {
                  rhs = connection.from.outputActivations.last!
               } else {
                  rhs = connection.from.outputActivations.first!
               }
               
               let innerProduct = connection.weights ⋅ rhs
               activations[portName]![0] = activations[portName]![0] + innerProduct
            }
         }
         if portName != "outputGates"{
            activations[portName]![0] = activations[portName]![0].map(port.activationFunc)
         }

      }
      
      

      states[0] = activations["inputGates"]![0] ⊙ activations["blockInputs"]![0] +
                  activations["forgetGates"]![0] ⊙ states[0]
      
      let outpGateActFunc = inputPorts["outputGates"]!.activationFunc
      let temp = activations["outputGates"]![0] + outputGate.peepholeWeights! ⊙ states[0]
      activations["outputGates"]?[0] = temp.map(outpGateActFunc)
      
      //TODO: change this from appending to inserting by index
      let outputActivation = activations["outputGates"]![0] ⊙ states[0].map(BrainKit.tanh)
      
      outputActivations.append(outputActivation)

      for t in 1..<inputCount{
         for (portName, port) in inputPorts{
            activations[portName]![t] = port.biases

            for connection in port.connections{
               let index = t + connection.offset
               let rhs:Vector
               if direction == .backward && connection.from is SourceLayer {
                  rhs = connection.from.outputActivations.reversed()[index]
               } else {
                  rhs = connection.from.outputActivations[index]
               }
               
               let innerProduct = connection.weights ⋅ rhs
               activations[portName]![t] = activations[portName]![t] + innerProduct
            }
            
            if portName != "outputGates"{
               if let peepholeWeights = port.peepholeWeights{
                  activations[portName]![t] = activations[portName]![t] + peepholeWeights ⊙ states[t-1]
               }
               activations[portName]![t] = activations[portName]![t].map(port.activationFunc)
               
            }

         }

         
         states[t] = activations["inputGates"]![t] ⊙ (activations["blockInputs"]?[t])! +
                    (activations["forgetGates"]![t] ⊙ states[t - 1])

         
         let temp = activations["outputGates"]![t] + outputGate.peepholeWeights! ⊙ states[t]
         activations["outputGates"]?[t] = temp.map(outpGateActFunc)

         let outputActivation = activations["outputGates"]![t] ⊙ states[t].map{BrainKit.tanh($0)}
         outputActivations.append(outputActivation)
         
         
      }
      if direction == .backward {
         self.outputActivations = self.outputActivations.reversed()
      }
      
   }
   
   func feedBack(){
      print("feeding back LSTM layer: \(self.name)")
//      for (_,inputPort) in inputPorts{
//         let Z = outputActivations[i] ⋅ outputErrors[i];
//         for j in 0..<outputActivations[i].count{
//            self.inputPort.errors?[i][j] = outputActivations[i][j] * (outputErrors[i][j] - Z)
//         }
//
//      }
//      self.outputErrors = self.inputErrors ⋅ self.biases
   }

   /**Calculates and returns the loss of the LSTM network.
    - Parameter input: The input to the network.
    - Parameter target: The target output, given as an array containing arrays of expected indexes.
    - numExamples: The number of examples used for training.
    - Returns: The loss of the network given the outputs and inputs.
    */
   open func calculateLoss(_ input: Matrix, target: [[Int]], numExamples: Int) -> Rational {
      //let result = feedForward(input)
      let result = Matrix()
      var L = [Rational]()
      var y = Matrix.zero(rows: result.count, cols: result[0].count)

      for i in 0..<target.count {
         for k in target[i] {
            y[i][k] = 1.0
         }
      }
      //
      for i in 0..<target.count {
         let product = y[i] ⊙ log(result[i])
         let _sum = product.sum()
         L.append(_sum)
      }
   
      return L.sum() / Rational(numExamples)
      
   }
}
