; implement factorial
;
;
; to build just run: make
;
; and debug with vscode this asm file
;
; Lesson links:
; https://en.wikibooks.org/wiki/X86_Assembly/Interfacing_with_Linux


%include "sys_codes.inc"
%include "unistd_64h.inc"


global _start                   ;must be declared for linker (gcc)

section .data
    strMsgWelcome:      db  "Please enter number:", 13, 10, 0   ; string
    strMsgWelcomeLen:   equ $-strMsgWelcome                     ; Length of string

    strMsgResults:      db  13, 10, "Number factorial:", 13, 10, 0  ; string
    strMsgResultsLen:   equ $-strMsgResults                     ; Length of string

    result_lp           dq 0                                    ; loop result
    result_rec          dq 0                                    ; recursion result

section .bss
    strInput            resb 1024                               ; reserve bytes
    strInputLen         equ $-strInput

; Code goes in text section
section .text

; can also use r8-r16 registers, but skip it for now...
_start:
    mov     rsi, strMsgWelcome      ; show welcome message for number input
    mov     rdx, strMsgWelcomeLen   ;
    call    print_string

    mov     rsi, strInput           ; read keyboard, fill chars in strInput, and rax it's count
    mov     rdx, strInputLen        ;
    call    Keyboard                ;

    mov     rsi, strInput           ; convert input string to number in rax
    mov     rcx, rax                ; number og diggits in strInput
    call    strtoi

    ; factorial loop
    push    rax                     ; save input number
    mov     rbx, rax                ; pass param from rax to rbx
    call    factorial_loop          ;

    mov     rsi, strInput           ; pass input params: rsi - string, rax already set
    call    inttostr
    mov     [result_lp], rax

    mov     rsi, strMsgResults      ; show results message
    mov     rdx, strMsgResultsLen   ;
    call    print_string

    mov     rsi, strInput           ; show factorial as string
    mov     rdx, rax                ;
    call    print_string
    pop     rax                     ; restote input number for next usage

    ; factorial recursive
    mov     rbx, rax                ; pass param from rax to rbx
    mov     rax, 1                  ; initial factorial value
    call    factorial_recursive     ;
    mov     [result_rec], rax

    mov     rsi, strInput           ; pass input params: rsi - string, rax already set
    call    inttostr

    mov     rsi, strMsgResults      ; show results message
    mov     rdx, strMsgResultsLen   ;
    call    print_string

    mov     rsi, strInput           ; show factorial as string
    mov     rdx, rax                ;
    call    print_string

    ;call program_exit
    mov     rdi, 0                  ; program exit resulting code 8-bit, error code or 0 etc
    mov     rax, __NR_exit          ; system call number (__NR_exit: 60)
    syscall                         ; call kernel


; Function: get input string
; Parameters:
;  rsi - output string
;  rdx - buffer length
get_user_input:
    mov     rax, __NR_read          ; system call command (__NR_read: 0)
    mov     rdi, fd_stdin           ; file descriptor (stdin)
    syscall                         ; call kernel
    ret


; Function: print string
; Parameters:
;  rsi - output string
;  rdx - string length
print_string:
    mov     rax, __NR_write     ; system call command (__NR_write: 1)
    mov     rdi, fd_stdout      ; file descriptor (stdout: 1)
    syscall                     ; call kernel
    ret


; Function:
; exit
; Parameters:
;  rdi - program result, error code (0 etc)
;
program_exit:
    ; Terminate program via linux exit
    mov     rax, __NR_exit      ; system call number (__NR_exit: 60)
    syscall                     ; call kernel
    ret


; Function.
;   number factorial
; Parameters:
;   rbx - number
; Returns:
;   rax
factorial_loop:
    mov     rax, 1              ; default
    cmp     rbx, 0
    push    rcx
    mov     rcx, rbx
    je      factorial_lb
    mov     rax, 1

    calc_fact_lp:
    mul     rcx
    loop calc_fact_lp

    pop     rcx
    factorial_lb:
    ret


; Function.
;   number factorial
; Parameters:
;   rbx - number
; Returns:
;   rax
factorial_recursive:
    cmp     rbx, 0
    mov     rcx, rbx
    je      factorial_rec_lb
    mul     rbx
    dec     rbx
    call    factorial_recursive

    factorial_rec_lb:
    ret


; Function.
;   string to number, checks every char if diggit
; Parameters:
;   rcx, rsi
; Returns:
;   rax
strtoi:
    cmp     rcx, 0              ; check if no diggits
    ja      has_chars_lb
    ;cmp     rcx, 1              ; check if 1 diggit
    ;ja      has_one_char_lb     ;
    mov     rax, 0              ; return zero
    ret                         ; first exit

    has_chars_lb:
    push    rsi
    push    rcx
    push    rdx

    mov     rdx, 0              ; temporary accumulation result
    ;dec     rcx                 ; -1 loop

    strtoi_lp:
    mov     bl , 10             ; base 10, pow(10, pos), rcx already has power of number
    dec     rcx                 ; -1 for power
    call    Power
    inc     rcx                 ;

    mov     bl, [rsi]
    cmp     bl, '0'
    jl      not_diggit_lb
    cmp     bl, '9'
    ja      not_diggit_lb
    sub     bl, '0'             ; single char to number
    push    rdx                 ; or use additional register
    mul     rbx                 ; mul rax to number
    pop     rdx
    add     rdx, rax            ; sum numbers
    inc     rsi
    loop strtoi_lp

    not_diggit_lb:              ; exit label
    mov     rax, rdx
    pop     rdx
    pop     rcx
    pop     rsi
    ret


; Function.
;   reverse string.
; Parameters:
;   rcx, rsi
; Returns:
;   rsi - reversed string
reverse_string:
    push    rax                 ; save used registers
    push    rdx
    mov     rdx, rcx            ; save input params
    mov     rax, rsi

    stackpush:
    mov     bl, [rsi]
    push    bx
    inc     rsi
    loop    stackpush

    mov     rcx, rdx
    mov     rsi, rax

    stackpop:
    pop     bx
    mov     [rsi], bl
    inc     rsi
    loop    stackpop

    mov     rcx, rdx            ; put return params
    mov     rsi, rax
    pop     rdx                 ; restore back used registers
    pop     rax
    ret


; Function.
;   number to string,
; Parameters:
;   rax, rsi
; Returns:
;   rsi
inttostr:
    push    rsi
    mov     rcx, 0              ; char counter
    inttostr_lp:
    mov     rdx, 0              ; clean rdx for div operation
    mov     ebx, 10              ; prepare divisor
    div     ebx
    add     dl, '0'
    mov     [rsi], dl
    inc     rsi
    inc     rcx
    cmp     rax, 0
    ja      inttostr_lp
    pop     rsi

    call reverse_string         ; rsi, rcx

    mov     rax, rcx            ; save length to rax  register
    ret


; Parameters:
;  rbx - number
;  rcx - power
; number to power
Power:
    ; intial checks for quick return
    cmp     rcx, 0              ; in power of 0, result is 1
    je      pow_is_zero_lbl
    cmp     rcx, 1              ; in power of 1, result is rax
    je      pow_is_one_lbl
    ; loop to find power of number
    push    rcx                 ; save rcx
    push    rdx                 ; save rdx
    mov     rax, rbx            ; store initial result same as input number, power of 1 same as
    dec     rcx
    pow_lp:
    mul     rbx
    loop pow_lp
    pop     rdx
    pop     rcx
    ret

    pow_is_zero_lbl:            ; second exit: if power 0, return 1
    mov     rax, 1
    ret

    pow_is_one_lbl:
    mov     rax, rbx            ; third exit, if power is 1, return initial number
    ret


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

    .Done	pop	rcx             ; Restore registers
        pop	rdi

         ret