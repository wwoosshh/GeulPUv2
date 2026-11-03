
C:\Codex\Agent\GeulNPU\鍮뚮뱶\npu\gl-aie\gl-packed-v3.o:	file format elf32-aie

Disassembly of section .text.geul_tile_packed:

00000000 <geul_tile_packed>:
       0: 7e 60 2b 00 0e 00 00 40 00 20 00 f0 2c 00    	nopa	;		nopb	;		paddxm	 [sp], #0x40;		nops	
       e: 98 15 d1 0f  	st	 r8, [sp, #-48]
      12: 98 35 d5 0f  	st	 r9, [sp, #-44]
      16: 98 55 d9 0f  	st	 r10, [sp, #-40]
      1a: 98 75 dd 0f  	st	 r11, [sp, #-36]
      1e: 98 95 e1 0f  	st	 r12, [sp, #-32]
      22: 0c 6b cb 0f a1 00    	mova	r1, #0x5;		st	 r13, [sp, #-28]
      28: 76 58 80 48 08 35 98 d5 e9 07 64 01  	mova	r4, #0xb;		st	 r14, [sp, #-24];		movx	r3, #0x328;		mov	r2, #0x80
      34: 76 78 60 32 2a 51 80 f5 ed 07 00 00  	mova	r0, #0x0;		st	 r15, [sp, #-20];		movx	r5, #0x9;		mov	p4, p2
      40: 76 58 07 e8 08 11 81 11 1c 04 93 00  	mova	r19, #0x4;		st	 r0, [p4], #4;		movx	r17, #0x8;		mov	r7, #0x7
      4c: 76 58 18 88 0a 52 81 11 1c 04 f6 ff  	mova	r22, #-0x1;		st	 r0, [p4], #4;		movx	r21, #0x10;		mov	r20, #0x18
      58: 76 20 00 00 00 00 80 11 1c 04 97 03  	mova	r23, #0x1c;		st	 r0, [p4], #4;		j	#0x0
      64: 76 10 00 c8 00 e0 bf 11 1c 04 58 01  	mova	r24, #0xa;		st	 r0, [p4], #4;		movxm	r6, #-0x800000
      70: 76 10 00 08 02 c0 bf 11 1c 04 19 fd  	mova	r25, #-0x18;		st	 r0, [p4], #4;		movxm	r16, #-0x1000000
      7c: 76 90 ff 4f fe 3f 80 11 1c 04 5a 00  	mova	r26, #0x2;		st	 r0, [p4], #4;		movxm	r18, #0xffffff
      88: 76 78 f0 b2 09 b7 bf 11 04 04 1c fe  	mova	r28, #-0x10;		st	 r0, [p4, #0];		movx	r27, #-0x8;		mov	p3, sp
      94: 76 58 00 a8 0b e0 81 11 14 f4 8c 7f  	padda	 [p3], #-0x40;		st	 r0, [p4, #4];		movx	r30, #0x0;		mov	r29, #0x0

000000a0 <.LBB0_1>:
      a0: 12 07 bc 27 00 f0 2c 00      	nopa	;		nopb	;		add	r30, r30, #0x1
      a8: 98 37 bf 17  	eq	 r31, r30, r19
      ac: 84 01 40 00 00 f8    	jnz	 r31, #0x0
      b2: 00 00        	nop	
      b4: 00 00        	nop	
      b6: 00 00        	nop	
      b8: 18 83 00 10  	add	r0, r0, #0x20
      bc: 18 63 30 16  	add	r24, r24, #0x18

000000c0 <.LBB0_2>:
      c0: e1 00 00 20 00 00 00 00 00 5b 01 20 00 f0 2c 00      	nopa	;		nopb	;		nops	;		j	#0x0;		nopv	
      d0: 18 00 00 10  	nopx	
      d4: 98 1d be 17  	lshl	 r31, r30, r1
      d8: e4 41 1f b1 04 fa    	or	 r8, r31, r2;		mov	dj0, r31
      de: f4 41 08 81 04 00    	vldb	 wl0, [p0, dj0];		mov	dj0, r8
      e4: b6 78 10 ee 0b 90 00 48 01 00 08 00  	mova	r8, #0x0;		vldb	 wl2, [p0, dj0];		movx	r9, #0x0;		mov	r31, r24

000000f0 <.LBB0_3>:
      f0: e4 81 cd 08 00 00    	nopx	;		mov	p4, p3
      f6: 98 56 1d 04  	lda	 r10, [p4], #4
      fa: 98 76 1d 04  	lda	 r11, [p4], #4
      fe: 00 00        	nop	
     100: 00 00        	nop	
     102: 00 00        	nop	
     104: 98 96 05 04  	lda	 r12, [p4, #0]
     108: 00 00        	nop	
     10a: 98 d6 15 04  	lda	 r14, [p4, #4]
     10e: 98 4d db 12  	lshl	 r13, r11, r20
     112: 98 bd d7 12  	lshl	 r11, r11, r27
     116: 98 a5 54 13  	or	 r10, r13, r10
     11a: 98 ad db 17  	lshl	 r13, r31, r26
     11e: 98 5d 1f 13  	lshl	 r15, r12, r21
     122: 98 cd 19 13  	lshl	 r12, r12, r28
     126: 98 1d 9d 13  	lshl	 r14, r14, r17
     12a: 98 c5 98 13  	or	 r12, r14, r12
     12e: 98 7c 5d 12  	ltu	 r14, r9, r23
     132: 98 b5 d6 13  	or	 r11, r15, r11
     136: 18 e3 5f 13  	add	r15, r13, #-0x8
     13a: 84 01 00 00 00 70    	jz	 r14, #0x0
     140: f8 a0 87 18  	mov	dj0, r15
     144: 5c e6 ab 36 2a 40    	st	 r10, [p2, dj0];		add	r10, r13, #-0x4
     14a: e4 41 0a 11 04 42    	add	 r8, r8, r2;		mov	dj0, r10
     150: 3a 79 50 43 78 f0 3f 30 2e 40	st	 r11, [p2, dj0];		add	r31, r31, #0x3;		mov	dj0, r13
     15a: 5c 26 a4 34 32 40    	st	 r12, [p2, dj0];		add	r9, r9, #0x4

00000160 <.LBB0_4>:
     160: 7e 60 2b 00 04 00 00 00 00 20 00 f0 2c 00    	nopa	;		nopb	;		j	#0x0;		nops	
		...
     176: ba 78 10 6a 85 a4 00 00 0c 00	mova	r12, #0x0;		add	 r10, r0, r9;		mov	r11, r8

00000180 <.LBB0_5>:
     180: 98 c0 9c 12  	add	 r14, r10, r12
     184: 98 9d 5f 13  	lshl	 r15, r13, r25
     188: 18 07 7a 17  	add	r29, r29, #0x1
     18c: 98 1d 9d 13  	lshl	 r14, r14, r17
     190: 98 f5 9c 13  	or	 r14, r14, r15
     194: 98 ad 5f 17  	lshl	 r15, r29, r26
     198: f8 a0 87 18  	mov	dj0, r15
     19c: 98 d1 01 0a  	st	 r14, [p2, dj0]

000001a0 <.LBB0_6>:
     1a0: e1 00 00 78 a5 01 6c ed 18 5b 01 20 00 f0 2c 00      	nopa	;		nopb	;		nops	;		lshl	 r14, r12, r26;		nopm	;		nopv	
     1b0: 18 07 18 13  	add	r12, r12, #0x1
     1b4: 98 37 1f 13  	eq	 r15, r12, r19
     1b8: 84 01 40 00 00 78    	jnz	 r15, #0x0
     1be: 00 00        	nop	
     1c0: 00 00        	nop	
     1c2: 00 00        	nop	
     1c4: e4 41 0e 91 64 6b    	and	 r13, r13, r18;		mov	dj0, r14
     1ca: 5c 06 ad 35 36 60    	st	 r13, [p3, dj0];		add	r11, r11, #0x20

000001d0 <.LBB0_7>:
     1d0: 02 70 d0 42 00 60 2b 00      	nops	;		mov	dj0, r11
     1d8: 18 24 01 39  	vldb	 wl4, [p1, dj0]
		...
     1e8: 08 81 e0 18  	vmul	dm0, x0, x4,r3
		...
     1f4: 00 00        	nop	
     1f6: f8 12 20 1b  	vmov	x6, bmll0
     1fa: f8 12 21 1c  	vmov	x8, bmlh0
     1fe: 22 81 e4 18 66 2d 43 03      	vadd.32	 x6, x8, x6;		vmul	dm0, x2, x4,r3
     206: 78 10 33 1c  	vshuffle	x8, x6, x6, r4
     20a: 78 2d 34 1b  	vadd.32	 x6, x6, x8
     20e: 78 14 33 1c  	vshuffle	x8, x6, x6, r5
     212: 78 2d 34 1a  	vadd.32	 x4, x6, x8
     216: 78 1c 22 1b  	vshuffle	x6, x4, x4, r7
     21a: f8 12 21 1c  	vmov	x8, bmlh0
     21e: 78 2d 23 1a  	vadd.32	 x4, x4, x6
     222: 78 04 22 1b  	vshuffle	x6, x4, x4, r1
     226: 78 2d 23 1a  	vadd.32	 x4, x4, x6
     22a: f8 12 20 1b  	vmov	x6, bmll0
     22e: 78 2d 43 1b  	vadd.32	 x6, x8, x6
     232: 78 10 33 1c  	vshuffle	x8, x6, x6, r4
     236: 78 2d 34 1b  	vadd.32	 x6, x6, x8
     23a: 78 14 33 1c  	vshuffle	x8, x6, x6, r5
     23e: 78 2d 34 1b  	vadd.32	 x6, x6, x8
     242: 78 1c 33 1c  	vshuffle	x8, x6, x6, r7
     246: 78 2d 34 1b  	vadd.32	 x6, x6, x8
     24a: 78 04 33 1c  	vshuffle	x8, x6, x6, r1
     24e: 78 2d 34 1b  	vadd.32	 x6, x6, x8
     252: b8 81 5a 1b  	vextract.32	 r13, x6, #0x0, vaddsign1
     256: b8 81 92 1b  	vextract.32	 r14, x4, #0x0, vaddsign1
     25a: 98 1d 5b 13  	lshl	 r13, r13, r17
     25e: 98 e0 5a 13  	add	 r13, r13, r14
     262: 98 60 5c 13  	add	 r14, r13, r6
     266: 98 0b 9d 13  	geu	 r14, r14, r16
     26a: 84 01 40 00 00 70    	jnz	 r14, #0x0
		...
     278: 00 00        	nop	
     27a: 98 7a 5c 17  	lt	 r14, r29, r7
     27e: 84 01 40 00 00 70    	jnz	 r14, #0x0
		...
     28c: 00 00        	nop	
     28e: 84 00 00 00 00 00    	j	#0x0
		...
     29c: 18 07 7a 17  	add	r29, r29, #0x1

000002a0 <.LBB0_10>:
     2a0: ba 00 00 00 00 00 00 20 be fd	lda	 r15, [sp, #-20];		nopxm	
     2aa: 18 d1 e9 07  	lda	 r14, [sp, #-24]
     2ae: 18 b1 e5 07  	lda	 r13, [sp, #-28]
     2b2: 18 91 e1 07  	lda	 r12, [sp, #-32]
     2b6: 18 71 dd 07  	lda	 r11, [sp, #-36]
     2ba: 18 51 d9 07  	lda	 r10, [sp, #-40]
     2be: 18 31 d5 07  	lda	 r9, [sp, #-44]
     2c2: 18 11 d1 07  	lda	 r8, [sp, #-48]
     2c6: 18 00 28 10  	ret	lr
     2ca: 00 00        	nop	
     2cc: 98 1a 77 17  	lt	 r27, r29, r17
     2d0: 18 6a 41 17  	sel.nez	 r0, r29, r22, r27
     2d4: 3a 71 00 00 00 fe 3f 30 82 40	st	 r0, [p2, #0];		paddxm	 [sp], #-0x40
     2de: 00 00        	nop	
