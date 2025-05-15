cmd    = ">>+>>>>>,[>+>>,]>+[--[+<<<-]<[<+>-]<[<[->[<<<+>>>>+<-]<<[>>+>[->]<<[<]<-]>]>>>+<[[-]<[>+<-]<]>[[>>>]+<<<-<[<<[<<<]>>+>[>>>]<-]<<[<<<]>[>>[>>>]<+<<[<<<]>-]]+<<<]+[->>>]>>]>>[.>>>]"
stdin_ = "41523"

import sys

TAPE_SIZE = 30_000


def build_bracemap(code: str) -> dict[int, int]:
    """Карта переходов между '[' и ']'."""
    stack, bm = [], {}
    for pos, c in enumerate(code):
        if c == '[':
            stack.append(pos)
        elif c == ']':
            if not stack:
                raise SyntaxError(f"Лишняя ']' на позиции {pos}")
            start = stack.pop()
            bm[start] = pos
            bm[pos] = start
    if stack:
        raise SyntaxError(f"Не закрыта '[' на позиции {stack.pop()}")
    return bm


def run(code: str, inp: str = "") -> None:
    tape = [0] * TAPE_SIZE
    ptr = ip = inp_ptr = 0
    bm = build_bracemap(code)

    while ip < len(code):
        c = code[ip]

        if c == '>':
            ptr += 1
        elif c == '<':
            ptr -= 1
        elif c == '+':
            tape[ptr] = (tape[ptr] + 1) & 0xFF
        elif c == '-':
            tape[ptr] = (tape[ptr] - 1) & 0xFF
        elif c == '.':
            sys.stdout.write(chr(tape[ptr]))
            sys.stdout.flush()
        elif c == ',':
            if inp_ptr < len(inp):
                tape[ptr] = ord(inp[inp_ptr])
                inp_ptr += 1
            else:
                tape[ptr] = 0
        elif c == '[' and tape[ptr] == 0:
            ip = bm[ip]
        elif c == ']' and tape[ptr] != 0:
            ip = bm[ip]

        ip += 1

        if not 0 <= ptr < TAPE_SIZE:
            raise IndexError("Указатель вышел за пределы ленты")


if __name__ == "__main__":
    run(cmd, stdin_)
    print()
