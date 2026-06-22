#!/usr/bin/env python3
"""Generate TECNOSALDO_Brainstorming_Brand.docx without external dependencies.

DOCX is an Open XML package (ZIP of XML files). This script builds one from
scratch using only the Python standard library.
"""

import zipfile
import os

RELS = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>"""

CONTENT_TYPES = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

WORD_RELS = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
    Target="styles.xml"/>
</Relationships>"""

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

STYLES = f"""\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:pPr><w:jc w:val="center"/></w:pPr>
    <w:rPr>
      <w:b/><w:sz w:val="56"/><w:szCs w:val="56"/>
      <w:color w:val="0D47A1"/>
      <w:rFonts w:ascii="Montserrat" w:hAnsi="Montserrat"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:pPr>
      <w:spacing w:before="360" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:b/><w:sz w:val="36"/><w:szCs w:val="36"/>
      <w:color w:val="0D47A1"/>
      <w:rFonts w:ascii="Montserrat" w:hAnsi="Montserrat"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:pPr>
      <w:spacing w:before="240" w:after="80"/>
    </w:pPr>
    <w:rPr>
      <w:b/><w:sz w:val="28"/><w:szCs w:val="28"/>
      <w:color w:val="0A0F17"/>
      <w:rFonts w:ascii="Montserrat" w:hAnsi="Montserrat"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:pPr>
      <w:spacing w:after="120" w:line="276" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:sz w:val="22"/><w:szCs w:val="22"/>
      <w:rFonts w:ascii="Montserrat" w:hAnsi="Montserrat"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:after="200"/>
    </w:pPr>
    <w:rPr>
      <w:i/><w:sz w:val="24"/><w:szCs w:val="24"/>
      <w:color w:val="6B7280"/>
      <w:rFonts w:ascii="Montserrat" w:hAnsi="Montserrat"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Quote">
    <w:name w:val="Quote"/>
    <w:pPr>
      <w:ind w:left="720"/>
      <w:spacing w:before="120" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:b/><w:sz w:val="26"/><w:szCs w:val="26"/>
      <w:color w:val="FFB000"/>
      <w:rFonts w:ascii="Montserrat" w:hAnsi="Montserrat"/>
    </w:rPr>
  </w:style>
</w:styles>"""


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _para(text: str, style: str = "Normal", bold: bool = False) -> str:
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    bold_xml = "<w:b/>" if bold else ""
    return (
        f'<w:p><w:pPr>{style_xml}</w:pPr>'
        f'<w:r><w:rPr>{bold_xml}</w:rPr>'
        f'<w:t xml:space="preserve">{_esc(text)}</w:t></w:r></w:p>'
    )


def _heading(text: str, level: int = 1) -> str:
    style = f"Heading{level}"
    return (
        f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
        f'<w:r><w:t xml:space="preserve">{_esc(text)}</w:t></w:r></w:p>'
    )


def _separator() -> str:
    return (
        '<w:p><w:pPr><w:pBdr>'
        '<w:bottom w:val="single" w:sz="6" w:space="1" w:color="6B7280"/>'
        '</w:pBdr></w:pPr></w:p>'
    )


def build_body() -> str:
    parts: list[str] = []

    # Title page
    parts.append(_para("TECNOSALDO", "Title"))
    parts.append(_para("Brainstorming Strategico del Brand", "Subtitle"))
    parts.append(_para("SALDIAMO IL PRESENTE. COSTRUIAMO IL FUTURO.", "Quote"))
    parts.append(_para(
        "Soluzioni, prodotti e competenza per la saldatura professionale. "
        "Affidabilità, innovazione e passione al servizio di ogni progetto."
    ))
    parts.append(_separator())

    sections = [
        ("11. BRAINSTORMING STRATEGICO DEL BRAND", [
            ("DNA DEL MARCHIO", [
                "Tecnosaldo è il partner tecnico che affianca aziende e professionisti "
                "della saldatura nella scelta delle migliori soluzioni per produttività, "
                "qualità e continuità operativa.",
            ]),
            ("MISSION", [
                "Fornire soluzioni professionali per la saldatura industriale attraverso "
                "prodotti affidabili, competenza tecnica e supporto continuo.",
            ]),
            ("VISION", [
                "Diventare il punto di riferimento in Liguria e nel Nord Italia per "
                "impianti, consumabili e servizi dedicati alla saldatura professionale.",
            ]),
        ]),
        ("12. BRAND POSITIONING", [
            (None, [
                "Non vendiamo semplicemente prodotti.",
                "Offriamo continuità produttiva, affidabilità, riduzione dei fermi macchina, "
                "sicurezza operativa, supporto tecnico specializzato e soluzioni complete.",
                "",
                "Il nostro posizionamento si fonda su:",
                "• Distributore ufficiale Miller Electric in Liguria",
                "• Assistenza tecnica specializzata pre e post-vendita",
                "• Gamma completa: impianti, consumabili, accessori, DPI, gas tecnici",
                "• Servizi a valore aggiunto: noleggio, formazione, consulenza",
            ]),
        ]),
        ("13. BRAND PERSONALITY", [
            (None, [
                "Il carattere del brand si esprime attraverso sei tratti distintivi:",
                "",
                "• Affidabili — Manteniamo ogni promessa",
                "• Competenti — Esperienza reale, risultati concreti",
                "• Concreti — Orientati alla soluzione, non alla teoria",
                "• Innovativi — Investiamo in tecnologia per il vantaggio del cliente",
                "• Disponibili — Sempre al fianco del professionista",
                "• Dinamici — Reattivi, flessibili, pronti a rispondere",
            ]),
        ]),
        ("14. TONE OF VOICE", [
            (None, [
                "La comunicazione Tecnosaldo è:",
                "",
                "• Professionale — Linguaggio da esperto a esperto",
                "• Chiaro — Nessuna ambiguità, messaggi diretti",
                "• Tecnico — Specifiche, dati, performance reali",
                "• Diretto — Orientato all'azione e alla soluzione",
                "• Orientato alla soluzione — Ogni comunicazione risponde a un bisogno concreto",
            ]),
        ]),
        ("15. TARGET DI RIFERIMENTO", [
            ("Target Primario", [
                "• Carpenterie metalliche",
                "• Officine meccaniche",
                "• Costruttori di impianti industriali",
                "• Industria manifatturiera pesante e leggera",
                "• Cantieri navali",
            ]),
            ("Target Secondario", [
                "• Saldatori specializzati e autonomi",
                "• Installatori e manutentori industriali",
                "• Centri di formazione professionale",
                "• Responsabili acquisti e operations manager",
            ]),
        ]),
        ("16. CUSTOMER JOURNEY", [
            (None, [
                "Il percorso del cliente Tecnosaldo si articola in cinque fasi:",
                "",
                "1. ANALISI — Comprensione delle esigenze produttive e operative del cliente",
                "2. CONSULENZA — Proposta tecnica personalizzata con soluzioni ottimali",
                "3. FORNITURA — Consegna di impianti, consumabili e accessori selezionati",
                "4. SUPPORTO — Assistenza tecnica continua, formazione e manutenzione",
                "5. FIDELIZZAZIONE — Partnership duratura basata su risultati e fiducia",
            ]),
        ]),
        ("17. PILASTRI DEL BRAND", [
            (None, [
                "I cinque pilastri su cui si fonda l'identità Tecnosaldo:",
                "",
                "• COMPETENZA TECNICA — Conoscenza approfondita dei processi di saldatura",
                "• ASSISTENZA — Supporto tecnico pre e post-vendita sempre disponibile",
                "• INNOVAZIONE — Tecnologie all'avanguardia per la massima produttività",
                "• AFFIDABILITÀ — Prodotti certificati, partner di fiducia",
                "• PARTNERSHIP — Relazioni di valore a lungo termine con ogni cliente",
            ]),
        ]),
        ("18. VALUE PROPOSITION", [
            (None, [
                "Tecnosaldo offre un ecosistema completo per la saldatura professionale:",
                "",
                "• Distributore ufficiale Miller Electric",
                "• Assistenza tecnica specializzata con personale qualificato",
                "• Formazione professionale e aggiornamento tecnico",
                "• Consulenza nella scelta di impianti e processi di saldatura",
                "• Noleggio impianti per cantieri e produzioni temporanee",
                "• Supporto post-vendita con ricambi e manutenzione programmata",
                "• Fornitura gas tecnici e consumabili di qualità",
            ]),
        ]),
        ("PAYOFF CONSIGLIATO", [
            (None, [
                "",
            ]),
        ]),
    ]

    for section_title, subsections in sections:
        parts.append(_heading(section_title, 1))
        for sub_title, paragraphs in subsections:
            if sub_title:
                parts.append(_heading(sub_title, 2))
            for p in paragraphs:
                if p == "":
                    parts.append(f'<w:p/>')
                else:
                    parts.append(_para(p))

    # Payoff in accent style
    parts.append(_para("IL PARTNER TECNICO DELLA SALDATURA PROFESSIONALE", "Quote"))
    parts.append(_para(""))

    # Color palette reference section
    parts.append(_separator())
    parts.append(_heading("RIFERIMENTI BRAND IDENTITY", 1))
    parts.append(_heading("Color Palette", 2))
    parts.append(_para("• #0A0F17 — Nero Blu (primario)"))
    parts.append(_para("• #0D47A1 — Blu Tecnico"))
    parts.append(_para("• #FFB000 — Giallo Energia"))
    parts.append(_para("• #6B7280 — Grigio Industriale"))
    parts.append(_para("• #F2F4F7 — Grigio Chiaro"))
    parts.append(_heading("Tipografia", 2))
    parts.append(_para("Font principale: Montserrat (Bold, SemiBold, Medium, Regular, Light)"))
    parts.append(_para("Font moderna, solida e leggibile. Trasmette affidabilità, forza e innovazione."))
    parts.append(_heading("Valori del Brand", 2))
    parts.append(_para("Affidabilità • Innovazione • Competenza • Precisione • Partnership • Sicurezza"))

    return "\n".join(parts)


def main():
    body = build_body()

    document_xml = f"""\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}">
  <w:body>
{body}
  </w:body>
</w:document>"""

    out_path = "/home/user/hand-gesture-control/TECNOSALDO_Brainstorming_Brand.docx"

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("word/_rels/document.xml.rels", WORD_RELS)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", STYLES)

    print(f"Created: {out_path}")
    print(f"Size: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    main()
