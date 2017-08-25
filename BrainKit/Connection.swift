//
//  Connection.swift
//  DSRKit
//
//  Created by Matthew Whiteside on 12/29/16.
//  Copyright © 2016 mattwhiteside. All rights reserved.
//

import Foundation
//import Upsurge

//extension Matrix where Element == Double{
//   func delayed() -> Matrix<Element>{
//      return self
//   }
//}


//struct Connection : NetworkNode{
//
//   let name:String
//   let from,to:Layer
//   let delays:Matrix<Double>
//   let weights:[Rational]
//   let weightOptimizerDeltas:[Rational]
//   let biases:[Rational]
//   let biasOptimizerDeltas:[Rational]
//   
//   init(name:String,
//        from:Layer,
//        to:Layer,
//        weights: [Rational],
//        weightOptimizerDeltas:[Rational],
//        biases:[Rational],
//        biasOptimizerDeltas:[Rational]){
//      self.name = name
//      self.from = from
//      self.to = to
//      self.delays = Matrix<Double>(rows: 0, columns: 0)
//      self.weights = weights
//      self.weightOptimizerDeltas = weightOptimizerDeltas
//      self.biases = biases
//      self.biasOptimizerDeltas = biasOptimizerDeltas
//   }
//   
//
//   
//
//   
//   func feedForward(inputVec:ValueArray<Rational>) -> Void {
//      var x = [Rational](repeating: 0, count: self.from.outputActivations.elements.count)
//      for i in 0..<self.from.outputActivations.elements.count{
//         x[i] = self.from.outputActivations.elements[i]
//      }
//      
//      let lhs = Matrix<Rational>(rows: 400,
//                                 columns: SymbolFeatureConfig.ON_FEAT,
//                                 elements: self.weights)
//      
//      let rhs:Matrix<Rational> = Matrix<Rational>(rows: SymbolFeatureConfig.ON_FEAT,
//                                                  columns: 1,
//                                                  elements: inputVec)
//      
//      print("doing something useful")
//      //compile error pick back up from here
//      let multResult = lhs * rhs
//      self.to.inputActivations = Tensor<Rational>(dimensions:[400,1], elements:multResult.elements)
//   }
//   
//   func feedBack() {
//      
//   }
//
//
//}
