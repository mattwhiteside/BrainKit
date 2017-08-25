//
//  BLSTMConfig.swift
//  BrainKit
//
//  Created by Matthew Whiteside on 1/30/17.
//  Copyright © 2017 mattwhiteside. All rights reserved.
//

import Foundation

public final class BLSTMConfig{
   let json:Dictionary<String, Any>
   init?(filepath:String){
      let url = URL(fileURLWithPath: filepath)
      do {
         let data = try Data(contentsOf: url)
         let jsonRaw = try JSONSerialization.jsonObject(with:data)
         if let _json = jsonRaw as? Dictionary<String, Any>{
            json = _json
         } else {
            return nil
         }
      } catch{
         fatalError("couldn't initialize config")
      }
   }
   
   subscript(key: String) -> Any{
      return json[key]!
   }
}
