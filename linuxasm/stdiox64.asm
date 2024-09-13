; https://www.cs.colostate.edu/~fsieker/misc/NumberConversion.html


section .data

    dataval dd 23

    ; power of 10 from 0 to 19 (max 64 bit value)
    ;power_of_10     dq  1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000, 10000000000000, 100000000000000, 1000000000000000, 10000000000000000, 100000000000000000, 1000000000000000000, 10000000000000000000


section .text




; ========================================================================================================
; Read keyboard entry until EOF (return) or overflow ENTRY > RCX.

;   ENTRY:  RDX = maximum characters
;           RSI = Pointer to input buffer

;   LEAVE:  If ZR = 0
;           RAX = Actuall size of entry
;           RDX = Unused space in this buffer
;           RSI = Pointer to next position
;
;           if ZR = 1
;               RAX = NULL
;               All others unchanged
;   NOTE:   Routine terminates strings with null to be "C" style conformant.
;                                       25H - 37 Bytes
; ---------------------------------------------------------------------------------------------------------

Keyboard:
        push	rdi
        push	rcx             ; Preserve

        xor	rax, rax            ; SYS_READ = 0
        mov	rdi, rax            ;    STDIN = 0
        syscall
        dec	rax                 ; Was anything other than carriage return entered
        jz	.Done               ; Bounce if RAX = NULL

    ; To facilitate entry within a loop I'm going to adjust RDX & RSI accordingly

        add	rsi, rax            ; Point to this entry's carriage return
        mov	byte [rsi], 0       ; Change to NULL
        inc	rsi                 ; Bump to next possible entry position
        sub	rdx, rax
        dec	rdx                 ; Represents room left in this buffer
        or	rax, rax            ; Reset flags

    .Done
        pop	rcx             ; Restore registers
        pop	rdi

         ret


