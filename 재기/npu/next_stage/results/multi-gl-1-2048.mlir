module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_1 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @a(%logical_shim_noc, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<256xi8>>  
    aie.objectfifo @b(%logical_shim_noc_0, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>  
    aie.objectfifo @c(%logical_core, {%logical_shim_noc_1}, 2 : i32) : !aie.objectfifo<memref<128xi32>>  
    func.func private @geul_tile(memref<256xi8>, memref<1024xi8>, memref<128xi32>) attributes {link_with = "C:\\Codex\\Agent\\GeulNPU\\\EB\B9\8C\EB\93\9C\\npu\\gl-aie\\gl.o"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_2 = arith.constant 0 : index
        %c2048 = arith.constant 2048 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c2048 step %c1_3 {
          %1 = aie.objectfifo.acquire @a(Consume, 1) : !aie.objectfifosubview<memref<256xi8>>
          %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>
          %3 = aie.objectfifo.acquire @b(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
          %5 = aie.objectfifo.acquire @c(Produce, 1) : !aie.objectfifosubview<memref<128xi32>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
          func.call @geul_tile(%2, %4, %6) : (memref<256xi8>, memref<1024xi8>, memref<128xi32>) -> ()
          aie.objectfifo.release @a(Consume, 1)
          aie.objectfifo.release @b(Consume, 1)
          aie.objectfifo.release @c(Produce, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<524288xi8>, %arg1: memref<2097152xi8>, %arg2: memref<262144xi32>) {
      %1 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%arg0 : memref<524288xi8>, 0, 524288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 524288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @b {
        aie.dma_bd(%arg1 : memref<2097152xi8>, 0, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @c {
        aie.dma_bd(%arg2 : memref<262144xi32>, 0, 262144, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 262144, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
    }
  }
}
