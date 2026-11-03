module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_0 = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_2 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_3 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_4 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @a(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<512xi8>>  
    aie.objectfifo @a0(%logical_mem, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<256xi8>>  
    aie.objectfifo @a1(%logical_mem, {%logical_core_0}, 2 : i32) : !aie.objectfifo<memref<256xi8>>  
    aie.objectfifo.link [@a] -> [@a0, @a1]([] [0, 256])
    aie.objectfifo @b(%logical_shim_noc_1, {%logical_mem_2}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>  
    aie.objectfifo @b0(%logical_mem_2, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>  
    aie.objectfifo @b1(%logical_mem_2, {%logical_core_0}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>  
    aie.objectfifo.link [@b] -> [@b0, @b1]([] [0, 1024])
    aie.objectfifo @c(%logical_mem_3, {%logical_shim_noc_4}, 2 : i32) : !aie.objectfifo<memref<256xi32>>  
    aie.objectfifo @c0(%logical_core, {%logical_mem_3}, 2 : i32) : !aie.objectfifo<memref<128xi32>>  
    aie.objectfifo @c1(%logical_core_0, {%logical_mem_3}, 2 : i32) : !aie.objectfifo<memref<128xi32>>  
    aie.objectfifo.link [@c0, @c1] -> [@c]([0, 128] [])
    func.func private @geul_tile(memref<256xi8>, memref<1024xi8>, memref<128xi32>) attributes {link_with = "C:\\Codex\\Agent\\GeulNPU\\\EB\B9\8C\EB\93\9C\\npu\\gl-aie\\gl.o"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_5 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c1024 step %c1_6 {
          %2 = aie.objectfifo.acquire @a0(Consume, 1) : !aie.objectfifosubview<memref<256xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>
          %4 = aie.objectfifo.acquire @b0(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
          %6 = aie.objectfifo.acquire @c0(Produce, 1) : !aie.objectfifosubview<memref<128xi32>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
          func.call @geul_tile(%3, %5, %7) : (memref<256xi8>, memref<1024xi8>, memref<128xi32>) -> ()
          aie.objectfifo.release @a0(Consume, 1)
          aie.objectfifo.release @b0(Consume, 1)
          aie.objectfifo.release @c0(Produce, 1)
        }
      }
      aie.end
    }
    %1 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_5 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg1 = %c0_5 to %c1024 step %c1_6 {
          %2 = aie.objectfifo.acquire @a1(Consume, 1) : !aie.objectfifosubview<memref<256xi8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>
          %4 = aie.objectfifo.acquire @b1(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
          %6 = aie.objectfifo.acquire @c1(Produce, 1) : !aie.objectfifosubview<memref<128xi32>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
          func.call @geul_tile(%3, %5, %7) : (memref<256xi8>, memref<1024xi8>, memref<128xi32>) -> ()
          aie.objectfifo.release @a1(Consume, 1)
          aie.objectfifo.release @b1(Consume, 1)
          aie.objectfifo.release @c1(Produce, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<524288xi8>, %arg1: memref<2097152xi8>, %arg2: memref<262144xi32>) {
      %2 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%arg0 : memref<524288xi8>, 0, 524288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 524288, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @b {
        aie.dma_bd(%arg1 : memref<2097152xi8>, 0, 2097152, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2097152, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @c {
        aie.dma_bd(%arg2 : memref<262144xi32>, 0, 262144, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 262144, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%4)
      aiex.dma_await_task(%4)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
    }
  }
}
