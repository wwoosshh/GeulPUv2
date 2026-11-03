module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_0 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_1 = aie.logical_tile<CoreTile>(?, ?)
    %logical_core_2 = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem_4 = aie.logical_tile<MemTile>(?, ?)
    %logical_mem_5 = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_6 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @a(%logical_shim_noc, {%logical_mem}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>  
    aie.objectfifo @a0(%logical_mem, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<256xi8>>  
    aie.objectfifo @a1(%logical_mem, {%logical_core_0}, 2 : i32) : !aie.objectfifo<memref<256xi8>>  
    aie.objectfifo @a2(%logical_mem, {%logical_core_1}, 2 : i32) : !aie.objectfifo<memref<256xi8>>  
    aie.objectfifo @a3(%logical_mem, {%logical_core_2}, 2 : i32) : !aie.objectfifo<memref<256xi8>>  
    aie.objectfifo.link [@a] -> [@a0, @a1, @a2, @a3]([] [0, 256, 512, 768])
    aie.objectfifo @b(%logical_shim_noc_3, {%logical_mem_4}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>  
    aie.objectfifo @b0(%logical_mem_4, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>  
    aie.objectfifo @b1(%logical_mem_4, {%logical_core_0}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>  
    aie.objectfifo @b2(%logical_mem_4, {%logical_core_1}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>  
    aie.objectfifo @b3(%logical_mem_4, {%logical_core_2}, 2 : i32) : !aie.objectfifo<memref<1024xi8>>  
    aie.objectfifo.link [@b] -> [@b0, @b1, @b2, @b3]([] [0, 1024, 2048, 3072])
    aie.objectfifo @c(%logical_mem_5, {%logical_shim_noc_6}, 2 : i32) : !aie.objectfifo<memref<416xi32>>  
    aie.objectfifo @c0(%logical_core, {%logical_mem_5}, 2 : i32) : !aie.objectfifo<memref<104xi32>>  
    aie.objectfifo @c1(%logical_core_0, {%logical_mem_5}, 2 : i32) : !aie.objectfifo<memref<104xi32>>  
    aie.objectfifo @c2(%logical_core_1, {%logical_mem_5}, 2 : i32) : !aie.objectfifo<memref<104xi32>>  
    aie.objectfifo @c3(%logical_core_2, {%logical_mem_5}, 2 : i32) : !aie.objectfifo<memref<104xi32>>  
    aie.objectfifo.link [@c0, @c1, @c2, @c3] -> [@c]([0, 104, 208, 312] [])
    func.func private @geul_tile_packed(memref<256xi8>, memref<1024xi8>, memref<104xi32>) attributes {link_with = "C:\\Codex\\Agent\\GeulNPU\\\EB\B9\8C\EB\93\9C\\npu\\gl-aie\\gl-packed.o"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_7 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c256 step %c1_8 {
          %4 = aie.objectfifo.acquire @b0(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
          %c0_9 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c4 step %c1_10 {
            %6 = aie.objectfifo.acquire @a0(Consume, 1) : !aie.objectfifosubview<memref<256xi8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>
            %8 = aie.objectfifo.acquire @c0(Produce, 1) : !aie.objectfifosubview<memref<104xi32>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<104xi32>> -> memref<104xi32>
            func.call @geul_tile_packed(%7, %5, %9) : (memref<256xi8>, memref<1024xi8>, memref<104xi32>) -> ()
            aie.objectfifo.release @a0(Consume, 1)
            aie.objectfifo.release @c0(Produce, 1)
          }
          aie.objectfifo.release @b0(Consume, 1)
        }
      }
      aie.end
    }
    %1 = aie.core(%logical_core_0) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_7 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c256 step %c1_8 {
          %4 = aie.objectfifo.acquire @b1(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
          %c0_9 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c4 step %c1_10 {
            %6 = aie.objectfifo.acquire @a1(Consume, 1) : !aie.objectfifosubview<memref<256xi8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>
            %8 = aie.objectfifo.acquire @c1(Produce, 1) : !aie.objectfifosubview<memref<104xi32>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<104xi32>> -> memref<104xi32>
            func.call @geul_tile_packed(%7, %5, %9) : (memref<256xi8>, memref<1024xi8>, memref<104xi32>) -> ()
            aie.objectfifo.release @a1(Consume, 1)
            aie.objectfifo.release @c1(Produce, 1)
          }
          aie.objectfifo.release @b1(Consume, 1)
        }
      }
      aie.end
    }
    %2 = aie.core(%logical_core_1) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_7 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c256 step %c1_8 {
          %4 = aie.objectfifo.acquire @b2(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
          %c0_9 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c4 step %c1_10 {
            %6 = aie.objectfifo.acquire @a2(Consume, 1) : !aie.objectfifosubview<memref<256xi8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>
            %8 = aie.objectfifo.acquire @c2(Produce, 1) : !aie.objectfifosubview<memref<104xi32>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<104xi32>> -> memref<104xi32>
            func.call @geul_tile_packed(%7, %5, %9) : (memref<256xi8>, memref<1024xi8>, memref<104xi32>) -> ()
            aie.objectfifo.release @a2(Consume, 1)
            aie.objectfifo.release @c2(Produce, 1)
          }
          aie.objectfifo.release @b2(Consume, 1)
        }
      }
      aie.end
    }
    %3 = aie.core(%logical_core_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_7 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c1_8 = arith.constant 1 : index
        scf.for %arg1 = %c0_7 to %c256 step %c1_8 {
          %4 = aie.objectfifo.acquire @b3(Consume, 1) : !aie.objectfifosubview<memref<1024xi8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi8>> -> memref<1024xi8>
          %c0_9 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg2 = %c0_9 to %c4 step %c1_10 {
            %6 = aie.objectfifo.acquire @a3(Consume, 1) : !aie.objectfifosubview<memref<256xi8>>
            %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<256xi8>> -> memref<256xi8>
            %8 = aie.objectfifo.acquire @c3(Produce, 1) : !aie.objectfifosubview<memref<104xi32>>
            %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<104xi32>> -> memref<104xi32>
            func.call @geul_tile_packed(%7, %5, %9) : (memref<256xi8>, memref<1024xi8>, memref<104xi32>) -> ()
            aie.objectfifo.release @a3(Consume, 1)
            aie.objectfifo.release @c3(Produce, 1)
          }
          aie.objectfifo.release @b3(Consume, 1)
        }
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<1048576xi8>, %arg1: memref<1048576xi8>, %arg2: memref<425984xi32>) {
      %4 = aiex.dma_configure_task_for @a {
        aie.dma_bd(%arg0 : memref<1048576xi8>, 0, 1048576, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1048576, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @b {
        aie.dma_bd(%arg1 : memref<1048576xi8>, 0, 1048576, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1048576, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @c {
        aie.dma_bd(%arg2 : memref<425984xi32>, 0, 425984, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 425984, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%6)
      aiex.dma_await_task(%6)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%5)
    }
  }
}
