//from http://codeforcaffeine.com/programming/swift-3-ordered-dictionary/
//by MICHAEL KYRIACOU

public struct OrderedDictionary<K: Hashable, V> {
   var keys = [K]()
   var dict = [K:V]()
   
   public var count: Int {
      return self.keys.count
   }
   
   public subscript(key: K) -> V? {
      get {
         return self.dict[key]
      }
      set(newValue) {
         if newValue == nil {
            self.dict.removeValue(forKey:key)
            self.keys = self.keys.filter {$0 != key}
         } else {
            let oldValue = self.dict.updateValue(newValue!, forKey: key)
            if oldValue == nil {
               self.keys.append(key)
            }
         }
      }
   }
}

extension OrderedDictionary: Sequence {
   public func makeIterator() -> AnyIterator<(K,V)> {
      var counter = 0
      return AnyIterator {
         guard counter<self.keys.count else {
            return nil
         }
         let key = self.keys[counter]
         let next:(K,V) = (key, self.dict[self.keys[counter]]!)
         counter += 1
         return next
      }
   }
}

extension OrderedDictionary: CustomStringConvertible {
   public var description: String {
      let isString = type(of: self.keys[0]) == String.self
      var result = "["
      for key in keys {
         result += isString ? "\"\(key)\"" : "\(key)"
         result += ": \(self[key]!), "
      }
      result = String(result.characters.dropLast(2))
      result += "]"
      return result
   }
}

extension OrderedDictionary: ExpressibleByDictionaryLiteral {
   public init(dictionaryLiteral elements: (K, V)...) {
      self.init()
      for (key, value) in elements {
         self[key] = value
      }
   }
}
