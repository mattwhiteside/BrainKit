//
//  Math.swift
//  BrainKit
//
//  Created by Matthew Whiteside on 2/19/17.
//  Copyright © 2017 mattwhiteside. All rights reserved.
//

import Darwin
import MultiLinAlg

func σ(_ x:Rational) -> Rational {
   let expMax = Rational.greatestFiniteMagnitude
   let expLimit = Darwin.log(expMax);

   if (x < expLimit) {
      if (x > -expLimit) {
         return 1.0 / (1.0 + exp(-x));
      }
      return 0;
   }
   return 1;
}

func tanh(_ x:Rational) -> Rational {
   return 2*σ(2.0*x) - 1
}
