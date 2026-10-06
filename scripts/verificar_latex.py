"""
verificar_latex.py — confere a integridade do documento LaTeX sem compilar.

Percorre o documento a partir de frontmatter/main.tex (caminhos relativos à raiz
do repositório, como no Overleaf), seguindo \\include e \\input, e confere:
  - arquivos de \\include, \\input e \\includegraphics que não existem;
  - \\ref, \\eqref, \\pageref e \\autoref sem \\label correspondente;
  - \\label duplicados;
  - chaves de \\cite* ausentes de frontmatter/references.bib;
  - \\begin/\\end desbalanceados em cada arquivo;
  - "Capítulo~\\ref{...}" apontando para um apêndice (e vice-versa);
  - caracteres de controle soltos no meio do texto.
Ignora comentários (% até o fim da linha) e o conteúdo de ambientes verbatim.
Com --listar, imprime também os arquivos visitados e as figuras usadas.

Uso:  python scripts/verificar_latex.py [--listar]
Sai com código 1 se houver algum problema.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = "frontmatter/main.tex"
BIB = "frontmatter/references.bib"
# Arquivos do frontmatter incluídos pelo nome curto
CURTOS = {"preamble", "capa", "folha_rosto", "folha_aprovacao", "epigrafe",
          "agradecimentos", "resumo", "abstract"}

RE_COMENT = re.compile(r"(?<!\\)%.*")
RE_VERB = re.compile(r"\\begin\{(verbatim|lstlisting|Verbatim)\}.*?\\end\{\1\}", re.S)
RE_INC = re.compile(r"\\(include|input)\{([^}]+)\}")
RE_FIG = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
RE_LABEL = re.compile(r"\\label\{([^}]+)\}")
RE_REF = re.compile(r"\\(ref|eqref|pageref|autoref)\{([^}]+)\}")
RE_CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}")
RE_ENV = re.compile(r"\\(begin|end)\{([^}]+)\}")
RE_CAP_REF = re.compile(r"(Cap[íi]tulos?|Ap[êe]ndices?)~\\ref\{([^}]+)\}")


def limpar(texto):
    texto = RE_VERB.sub("", texto)
    return "\n".join(RE_COMENT.sub("", linha) for linha in texto.split("\n"))


def resolver(alvo):
    if alvo in CURTOS:
        alvo = "frontmatter/" + alvo
    return alvo if alvo.endswith(".tex") else alvo + ".tex"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true")
    args = ap.parse_args()

    problemas = []
    visitados, figuras = [], []
    labels = {}            # rótulo -> (arquivo, apêndice?)
    refs, cites, cap_refs = [], [], []

    apendice = [False]

    def visitar(caminho, origem):
        if not os.path.exists(os.path.join(ROOT, caminho)):
            problemas.append(f"arquivo ausente: {caminho} (em {origem})")
            return
        visitados.append(caminho)
        with open(os.path.join(ROOT, caminho), encoding="utf-8-sig", newline="") as fh:
            bruto = fh.read()
        # Caracteres de controle soltos (ex.: "\r" de um \ref mal escapado vira CR + "ef{")
        for m in re.finditer(r"\r(?!\n)|[\x00-\x08\x0b\x0c\x0e-\x1f]", bruto):
            linha = bruto.count("\n", 0, m.start()) + 1
            problemas.append(f"caractere de controle {m.group()!r}: {caminho}:{linha}")
        texto = limpar(bruto.replace("\r\n", "\n"))
        # Eventos na ordem do texto, para saber quando começa o \appendix
        eventos = []
        for rx, tipo in [(RE_INC, "inc"), (RE_FIG, "fig"), (RE_LABEL, "label"),
                         (RE_REF, "ref"), (RE_CITE, "cite"), (RE_CAP_REF, "capref")]:
            eventos += [(m.start(), tipo, m) for m in rx.finditer(texto)]
        eventos += [(m.start(), "appendix", m) for m in re.finditer(r"\\appendix\b", texto)]
        for _, tipo, m in sorted(eventos, key=lambda e: e[0]):
            if tipo == "appendix":
                apendice[0] = True
            elif tipo == "inc":
                visitar(resolver(m.group(2)), caminho)
            elif tipo == "fig":
                figuras.append(m.group(1))
                if not os.path.exists(os.path.join(ROOT, m.group(1))):
                    problemas.append(f"figura ausente: {m.group(1)} (em {caminho})")
            elif tipo == "label":
                lab = m.group(1)
                if lab in labels:
                    problemas.append(f"rótulo duplicado: {lab} ({labels[lab][0]} e {caminho})")
                labels[lab] = (caminho, apendice[0])
            elif tipo == "ref":
                refs.append((m.group(2), caminho))
            elif tipo == "cite":
                cites.extend((k.strip(), caminho) for k in m.group(1).split(","))
            elif tipo == "capref":
                cap_refs.append((m.group(1), m.group(2), caminho))
        pilha = []
        for m in RE_ENV.finditer(texto):
            linha = texto.count("\n", 0, m.start()) + 1
            if m.group(1) == "begin":
                pilha.append((m.group(2), linha))
            elif not pilha or pilha[-1][0] != m.group(2):
                problemas.append(f"\\end{{{m.group(2)}}} sem \\begin correspondente: {caminho}:{linha}")
            else:
                pilha.pop()
        for env, linha in pilha:
            problemas.append(f"\\begin{{{env}}} sem \\end: {caminho}:{linha}")

    visitar(MAIN, "-")

    for lab, onde in refs:
        if lab not in labels:
            problemas.append(f"\\ref sem \\label: {lab} (em {onde})")
    for palavra, lab, onde in cap_refs:
        if lab not in labels or not lab.startswith("chap:") and not lab.startswith("app:"):
            continue
        eh_apendice = labels[lab][1]
        if palavra.startswith("Cap") and eh_apendice:
            problemas.append(f"'{palavra}~\\ref{{{lab}}}' aponta para apêndice (em {onde})")
        if palavra.startswith("Ap") and not eh_apendice:
            problemas.append(f"'{palavra}~\\ref{{{lab}}}' aponta para capítulo (em {onde})")

    with open(os.path.join(ROOT, BIB), encoding="utf-8-sig") as fh:
        chaves = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", fh.read()))
    for chave, onde in cites:
        if chave not in chaves:
            problemas.append(f"citação ausente do .bib: {chave} (em {onde})")

    if args.listar:
        print("Arquivos visitados:")
        print("\n".join("  " + v for v in visitados))
        print("Figuras:")
        print("\n".join("  " + f for f in figuras))
    print(f"{len(visitados)} arquivos, {len(labels)} rótulos, {len(refs)} referências, "
          f"{len(figuras)} figuras, {len(cites)} citações ({len(set(c for c, _ in cites))} chaves)")
    if problemas:
        print(f"{len(problemas)} problema(s):")
        print("\n".join("  - " + p for p in problemas))
        sys.exit(1)
    print("OK: nenhum problema encontrado.")


if __name__ == "__main__":
    main()
