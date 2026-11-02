"""Generate fake car-dealership documents for development and evaluation.

Every person, plate, VIN, company and number here is fictional (no real customer data,
per RA 10173). Output: samples/*.pdf plus samples/manifest.json, which seed.py reads.

Most documents are clean PDFs. The Nissan Almera's papers exist only as a phone photo
(tilted, shadowed), a sideways low-contrast photo, and a skewed image-only scan, to
exercise OCR the way real dealership uploads would.

    python scripts/make_sample_docs.py
"""

import io
import json
import random
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageEnhance, ImageFilter

OUT_DIR = Path(__file__).resolve().parents[1] / "samples"
FOOTER = "SAMPLE DOCUMENT - FICTIONAL DATA FOR SOFTWARE TESTING ONLY"

INSURER = "Sampaguita Mutual Insurance Corp."
BANK = "Bayan Pag-asa Savings Bank"
INSPECTOR_SHOP = "Kalye Auto Inspection Center"
NOTARY = "Atty. Teodoro B. Navarro"

_rng = random.Random(1173)  # fixed seed: same documents every run
_VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


def _vin(prefix: str) -> str:
    return prefix + "".join(_rng.choice(_VIN_CHARS) for _ in range(17 - len(prefix)))


def _digits(n: int) -> str:
    return "".join(_rng.choice("0123456789") for _ in range(n))


VEHICLES = [
    dict(
        plate="NBC 1234",
        make="Toyota",
        model="Vios",
        variant="1.3 E MT",
        year=2019,
        color="Silver Metallic",
        status="sold",
        vin=_vin("MR0FB8CD"),
        engine="2NR" + _digits(7),
        owner="Juan P. Dela Cruz",
        address="12 Mabini St., Brgy. San Isidro, Quezon City",
        lto_office="Quezon City District Office",
    ),
    dict(
        plate="ABC 5678",
        make="Honda",
        model="City",
        variant="1.5 VX CVT",
        year=2020,
        color="Platinum White Pearl",
        status="listed",
        vin=_vin("MRHGM6"),
        engine="L15Z" + _digits(7),
        owner="Roberto C. Villanueva",
        address="45 Kapitolyo Rd., Pasig City",
        lto_office="Pasig District Office",
    ),
    dict(
        plate="DAE 9012",
        make="Mitsubishi",
        model="Mirage G4",
        variant="GLX 1.2 MT",
        year=2018,
        color="Red Metallic",
        status="sold",
        vin=_vin("MMBSNA13"),
        engine="3A92" + _digits(6),
        owner="Ana Marie T. Reyes",
        address="88 Osmena Blvd., Cebu City",
        lto_office="Cebu City District Office",
    ),
    dict(
        plate="NFQ 3456",
        make="Toyota",
        model="Fortuner",
        variant="2.4 G 4x2 AT",
        year=2021,
        color="Attitude Black",
        status="acquired",
        vin=_vin("MR0HA3FS"),
        engine="2GD" + _digits(7),
        owner="Liza G. Fernandez",
        address="7 Quimpo Blvd., Davao City",
        lto_office="Davao City District Office",
    ),
    dict(
        plate="CAR 7890",
        make="Ford",
        model="Ranger",
        variant="2.2 XLT 4x2 MT",
        year=2017,
        color="Aurora Blue",
        status="sold",
        vin=_vin("MNCUMFF8"),
        engine="QJ2R" + _digits(6),
        owner="Ramon D. Aquino",
        address="23 P. Burgos St., Batangas City",
        lto_office="Batangas District Office",
    ),
    dict(
        plate="PQR 2468",
        make="Suzuki",
        model="Ertiga",
        variant="GL 1.5 AT",
        year=2022,
        color="Magma Gray",
        status="listed",
        vin=_vin("MA3BNC32"),
        engine="K15B" + _digits(6),
        owner="Grace S. Mendoza",
        address="5 Rizal Ave., San Fernando, Pampanga",
        lto_office="San Fernando District Office",
    ),
]
# Only photographed/scanned papers exist for this car. Fixed values (no _rng calls) so
# adding it doesn't change the other generated documents.
VEHICLES.append(
    dict(
        plate="TAV 4521",
        make="Nissan",
        model="Almera",
        variant="1.0 Turbo VE CVT",
        year=2021,
        color="Brilliant Silver",
        status="sold",
        vin="MDHBN7AD5MJ602847",
        engine="HRA0551287",
        owner="Teresita V. Ocampo",
        address="16 Magsaysay Ave., Naga City",
        lto_office="Naga City District Office",
    )
)
V = {v["plate"]: v for v in VEHICLES}


def _vehicle_block(v: dict) -> str:
    return (
        f"Make: {v['make']}\nModel/Series: {v['model']} {v['variant']}\n"
        f"Year Model: {v['year']}\nColor: {v['color']}\nPlate No.: {v['plate']}\n"
        f"Engine No.: {v['engine']}\nChassis No. / VIN: {v['vin']}"
    )


# ---------------------------------------------------------------- OR/CR (2 pages)


def or_cr(v: dict, mv_file: str, or_no: str, or_date: str, amount: str, expiry: str) -> list:
    cr = (
        f"LAND TRANSPORTATION OFFICE - {v['lto_office']}\n\n"
        f"MV File No.: {mv_file}\nCR No.: {_digits(8)}\n\n"
        f"Owner: {v['owner'].upper()}\nAddress: {v['address']}\n\n"
        f"{_vehicle_block(v)}\n"
        "Body Type: " + ("Pick-up" if v["model"] == "Ranger" else "Sedan/SUV") + "\n"
        "Fuel: " + ("Diesel" if v["model"] in ("Fortuner", "Ranger") else "Gasoline") + "\n"
        "Classification: Private\n\n"
        "Encumbered to: " + (BANK if v["plate"] in ("NFQ 3456", "CAR 7890") else "None")
    )
    orec = (
        f"LAND TRANSPORTATION OFFICE - {v['lto_office']}\n\n"
        f"OR No.: {or_no}\nDate: {or_date}\nReceived from: {v['owner'].upper()}\n"
        f"Plate No.: {v['plate']}\nMV File No.: {mv_file}\n\n"
        "Particulars:\n"
        "  MVUC (Motor Vehicle User's Charge) ... PHP 1,600.00\n"
        "  Computer fee ........................ PHP 169.00\n"
        "  Legal research fund ................. PHP 10.00\n"
        f"  Other fees .......................... PHP {amount}\n\n"
        f"Registration valid until: {expiry}\n"
        "Payment mode: Cash"
    )
    return [("CERTIFICATE OF REGISTRATION (CR)", cr), ("OFFICIAL RECEIPT (OR)", orec)]


# ---------------------------------------------------------------- Deed of sale (2 pages)


def deed_of_sale_en(
    v: dict, buyer: str, buyer_addr: str, price: str, price_words: str, date: str, place: str
) -> list:
    p1 = (
        "KNOW ALL MEN BY THESE PRESENTS:\n\n"
        f"This Deed of Absolute Sale is made and executed on {date} at {place}, by and between:\n\n"
        f"{v['owner'].upper()}, of legal age, Filipino, residing at {v['address']}, "
        "hereinafter called the SELLER;\n\n- and -\n\n"
        f"{buyer.upper()}, of legal age, Filipino, residing at {buyer_addr}, "
        "hereinafter called the BUYER.\n\n"
        "The SELLER is the registered owner of the following motor vehicle:\n\n"
        f"{_vehicle_block(v)}\n\n"
        f"For and in consideration of the sum of {price_words} (PHP {price}), receipt of which "
        "is hereby acknowledged in full, the SELLER hereby sells, transfers and conveys the said "
        "motor vehicle to the BUYER, free from all liens and encumbrances.\n\n"
        "The SELLER warrants that the vehicle has no pending traffic violations, and that all "
        "taxes and registration fees up to the date of sale have been paid. The BUYER shall "
        "shoulder the transfer of ownership fees at the LTO."
    )
    p2 = (
        "IN WITNESS WHEREOF, the parties have signed this instrument on the date and place "
        "first written above.\n\n"
        f"______________________\n{v['owner'].upper()}\nSeller\n\n"
        f"______________________\n{buyer.upper()}\nBuyer\n\n"
        "Signed in the presence of:\n  Elena R. Cruz            Marco V. Lim\n\n"
        "ACKNOWLEDGMENT\n\n"
        f"REPUBLIC OF THE PHILIPPINES ) S.S.\n{place.upper()} )\n\n"
        f"Before me, a Notary Public, on {date}, personally appeared the above-named parties, "
        "known to me to be the same persons who executed the foregoing instrument, and they "
        "acknowledged that the same is their free and voluntary act and deed.\n\n"
        f"{NOTARY}\nNotary Public\nDoc. No. {_rng.randint(100, 499)}; Page No. "
        f"{_rng.randint(10, 99)}; Book No. XII; Series of {date[-4:]}."
    )
    return [
        ("DEED OF ABSOLUTE SALE OF MOTOR VEHICLE", p1),
        ("DEED OF ABSOLUTE SALE (continued)", p2),
    ]


def deed_of_sale_taglish(
    v: dict, buyer: str, buyer_addr: str, price: str, date: str, place: str
) -> list:
    p1 = (
        "ALAMIN NG LAHAT SA PAMAMAGITAN NITO:\n\n"
        f"Ang kasulatang ito ay ginawa noong {date} sa {place}, sa pagitan nina:\n\n"
        f"{v['owner'].upper()}, nasa hustong gulang, Pilipino, nakatira sa {v['address']}, "
        "na tatawaging NAGBEBENTA (Seller);\n\n- at -\n\n"
        f"{buyer.upper()}, nasa hustong gulang, Pilipino, nakatira sa {buyer_addr}, "
        "na tatawaging BUMIBILI (Buyer).\n\n"
        "Ang NAGBEBENTA ang rehistradong may-ari ng sasakyang ito:\n\n"
        f"{_vehicle_block(v)}\n\n"
        f"Ang napagkasunduang presyo ay PHP {price}, na buong natanggap na ng NAGBEBENTA. "
        "Dahil dito, ibinebenta at inililipat na ng NAGBEBENTA ang sasakyan sa BUMIBILI, "
        "walang anumang utang o encumbrance.\n\n"
        "Ang BUMIBILI ang sasagot sa transfer of ownership fees sa LTO. Ang NAGBEBENTA naman "
        "ang responsable sa anumang traffic violation bago ang petsa ng bentahan. "
        "Kasama sa benta ang spare tire, early warning device, at isang (1) duplicate key."
    )
    p2 = (
        "BILANG PATUNAY, lumagda ang magkabilang panig sa petsa at lugar na nakasaad sa itaas.\n\n"
        f"______________________\n{v['owner'].upper()}\nNagbebenta\n\n"
        f"______________________\n{buyer.upper()}\nBumibili\n\n"
        "Mga saksi:\n  Josefina A. Ramos            Paolo C. Tan\n\n"
        "ACKNOWLEDGMENT\n\n"
        f"Sa harap ko, isang Notary Public, noong {date}, ay personal na humarap ang mga "
        "partido na nabanggit sa itaas at kinilala nila na ang kasulatang ito ay kusang-loob "
        "nilang ginawa.\n\n"
        f"{NOTARY}\nNotary Public\nDoc. No. {_rng.randint(100, 499)}; Page No. "
        f"{_rng.randint(10, 99)}; Book No. VII; Series of {date[-4:]}."
    )
    return [
        ("KASULATAN NG GANAP NA BILIHAN NG SASAKYAN (Deed of Absolute Sale)", p1),
        ("KASULATAN NG BILIHAN (karugtong)", p2),
    ]


# ---------------------------------------------------------------- Insurance (2 pages)


def insurance(
    v: dict,
    policy_no: str,
    start: str,
    end: str,
    sum_insured: str,
    premium: str,
    deductible: str,
    aon: bool,
    mortgagee: str | None,
) -> list:
    aon_limit = f"PHP {sum_insured}" if aon else "Not covered"
    p1 = (
        f"{INSURER}\nMOTOR CAR POLICY - COMPREHENSIVE\n\n"
        f"Policy No.: {policy_no}\nAssured: {v['owner'].upper()}\nAddress: {v['address']}\n"
        f"Period of Insurance: {start} to {end} (12:00 noon)\n\n"
        "SCHEDULED VEHICLE\n"
        f"{_vehicle_block(v)}\n\n"
        "COVERAGE AND LIMITS\n"
        f"  Own Damage / Theft ................. PHP {sum_insured}\n"
        "  Excess Bodily Injury ............... PHP 200,000.00\n"
        "  Third Party Property Damage ........ PHP 200,000.00\n"
        "  Auto Personal Accident ............. PHP 50,000.00 per seat (5 seats)\n"
        f"  Acts of Nature (AON) ............... {aon_limit}\n\n"
        f"Total Premium (incl. taxes): PHP {premium}\n"
        f"Mortgagee: {mortgagee or 'None'}"
    )
    p2 = (
        "POLICY CONDITIONS (summary)\n\n"
        f"1. Deductible: PHP {deductible} per Own Damage claim (participation fee).\n"
        "2. Authorized drivers: the Assured, or any person driving on the Assured's order "
        "or with permission, holding a valid LTO driver's license.\n"
        "3. Claims must be reported within 72 hours of the incident, with a police report "
        "for theft and third-party claims.\n"
        "4. Exclusions: use for hire or reward, racing, driving under the influence, "
        "wear and tear, mechanical breakdown.\n"
        + (
            "5. Acts of Nature covers flood, typhoon, earthquake and volcanic eruption.\n"
            if aon
            else "5. Loss or damage from flood or typhoon is NOT covered under this "
            "policy (no AON rider).\n"
        )
        + "6. Towing: up to PHP 2,000.00 per incident to the nearest accredited repair shop.\n\n"
        "Claims hotline: (02) 8555-0199 (fictional)\n"
        f"Issued at Makati City on {start}."
    )
    return [("MOTOR CAR INSURANCE POLICY SCHEDULE", p1), ("POLICY CONDITIONS", p2)]


# ---------------------------------------------------------------- Inspection (1 page)


def inspection_en(
    v: dict,
    report_no: str,
    date: str,
    odometer: str,
    inspector: str,
    findings: list[str],
    rating: str,
    recommendation: str,
) -> list:
    body = (
        f"{INSPECTOR_SHOP}\nReport No.: {report_no}\nDate of inspection: {date}\n"
        f"Inspector: {inspector}\n\n{_vehicle_block(v)}\nOdometer reading: {odometer} km\n\n"
        "CHECKLIST (OK / ATTENTION / FAIL)\n"
        "  Engine (start, idle, leaks) .......... OK\n"
        "  Transmission ......................... OK\n"
        "  Brakes ............................... ATTENTION\n"
        "  Suspension / steering ................ OK\n"
        "  Electrical / lights .................. OK\n"
        "  Body / paint ......................... ATTENTION\n"
        "  Flood damage signs ................... None found\n"
        "  Underchassis rust .................... Minor\n\n"
        "FINDINGS\n" + "\n".join(f"  - {f}" for f in findings) + "\n\n"
        f"Overall condition rating: {rating}\nRecommendation: {recommendation}"
    )
    return [("PRE-PURCHASE VEHICLE INSPECTION REPORT", body)]


def inspection_taglish(
    v: dict,
    report_no: str,
    date: str,
    odometer: str,
    inspector: str,
    notes: list[str],
    rating: str,
    recommendation: str,
) -> list:
    body = (
        f"{INSPECTOR_SHOP}\nReport No.: {report_no}\nPetsa ng inspection: {date}\n"
        f"Inspector: {inspector}\n\n{_vehicle_block(v)}\nOdometer: {odometer} km\n\n"
        "CHECKLIST\n"
        "  Makina ............................... OK, walang leak\n"
        "  Transmission ......................... OK, smooth ang shifting\n"
        "  Preno (brakes) ....................... OK\n"
        "  Ilaw at electrical ................... May sira (tingnan sa notes)\n"
        "  Gulong (tires) ....................... Kailangan palitan\n"
        "  Senyales ng baha (flood) ............. Wala\n\n"
        "NOTES NG INSPECTOR\n" + "\n".join(f"  - {n}" for n in notes) + "\n\n"
        f"Overall rating: {rating}\nRekomendasyon: {recommendation}"
    )
    return [("VEHICLE INSPECTION REPORT", body)]


# ---------------------------------------------------------------- Financing (2 pages)


def financing_en(
    v: dict,
    loan_no: str,
    price: str,
    down: str,
    principal: str,
    rate: str,
    term: int,
    monthly: str,
    first_due: str,
    release: str,
) -> list:
    p1 = (
        f"{BANK}\nAuto Loan Division\n\nLoan Account No.: {loan_no}\n"
        f"Borrower: {v['owner'].upper()}\nAddress: {v['address']}\n"
        f"Date of release: {release}\n\nCOLLATERAL\n{_vehicle_block(v)}\n\n"
        "LOAN TERMS\n"
        f"  Cash price of vehicle .......... PHP {price}\n"
        f"  Down payment ................... PHP {down}\n"
        f"  Amount financed (principal) .... PHP {principal}\n"
        f"  Add-on interest rate ........... {rate} per annum\n"
        f"  Term ........................... {term} months\n"
        f"  Monthly amortization ........... PHP {monthly}\n"
        f"  First due date ................. {first_due}\n"
        "  Payment mode ................... Post-dated checks"
    )
    p2 = (
        "TERMS AND CONDITIONS (summary)\n\n"
        "1. The vehicle is mortgaged to the Bank under a Chattel Mortgage until the loan is "
        "fully paid. The OR/CR shall be annotated accordingly.\n"
        "2. Late payment penalty: 3% of the unpaid amortization per month of delay.\n"
        "3. Pre-termination fee: 4% of the outstanding principal balance.\n"
        "4. The Borrower must keep the vehicle insured (comprehensive) with the Bank as "
        "mortgagee for the full loan term.\n"
        "5. The vehicle may not be sold or transferred without the Bank's written consent.\n\n"
        f"Conforme:\n______________________\n{v['owner'].upper()}\nBorrower"
    )
    return [("AUTO LOAN DISCLOSURE STATEMENT", p1), ("LOAN TERMS AND CONDITIONS", p2)]


def financing_taglish(
    v: dict,
    loan_no: str,
    principal: str,
    balance: str,
    monthly: str,
    paid: int,
    term: int,
    as_of: str,
) -> list:
    p1 = (
        f"{BANK}\nAuto Loan Division\n\nLoan Account No.: {loan_no}\n"
        f"Borrower: {v['owner'].upper()}\nPetsa: {as_of}\n\n{_vehicle_block(v)}\n\n"
        "STATUS NG LOAN\n"
        f"  Orihinal na principal ............ PHP {principal}\n"
        f"  Buwanang hulog (monthly) ......... PHP {monthly}\n"
        f"  Nabayarang hulog ................. {paid} sa {term} months\n"
        f"  Natitirang balance (as of {as_of}) ... PHP {balance}\n\n"
        "PAALALA: Para mailipat ang sasakyan sa bagong may-ari, kailangang bayaran muna ang "
        "buong natitirang balance. Pagkatapos nito, mag-iissue ang Bank ng Release of Chattel "
        "Mortgage sa loob ng 15 banking days."
    )
    p2 = (
        "MGA KONDISYON\n\n"
        "1. May penalty na 3% kada buwan sa bawat hulog na late.\n"
        "2. Pre-termination fee: 4% ng natitirang principal. Pero kung ang full payment ay "
        "dahil sa bentahan ng sasakyan at may approval ng Bank, waived ang pre-termination fee.\n"
        "3. Ang original OR/CR ay hawak ng Bank hanggang fully paid ang loan.\n"
        "4. Kailangang may comprehensive insurance ang sasakyan habang may loan.\n\n"
        "Para sa mga tanong, tumawag sa (02) 8555-0142 (fictional) o mag-email sa "
        "autoloans@pagasa-bank.example."
    )
    return [("LOAN BALANCE STATEMENT (Statement of Account)", p1), ("MGA KONDISYON NG LOAN", p2)]


# ---------------------------------------------------------------- Document list


def build_documents() -> list[dict]:
    """Each entry: file, plate, doc_type, language, pages=[(title, body), ...]."""
    d = []

    def add(file, plate, doc_type, language, pages):
        d.append(dict(file=file, plate=plate, doc_type=doc_type, language=language, pages=pages))

    add(
        "nbc1234_orcr.pdf",
        "NBC 1234",
        "or_cr",
        "en",
        or_cr(
            V["NBC 1234"],
            "1301-00000482615",
            "8820391457",
            "March 14, 2024",
            "41.00",
            "March 31, 2025",
        ),
    )
    add(
        "abc5678_orcr.pdf",
        "ABC 5678",
        "or_cr",
        "en",
        or_cr(
            V["ABC 5678"],
            "1336-00000719342",
            "8820558104",
            "August 2, 2024",
            "56.00",
            "August 31, 2025",
        ),
    )
    add(
        "dae9012_orcr.pdf",
        "DAE 9012",
        "or_cr",
        "en",
        or_cr(
            V["DAE 9012"],
            "0701-00000358127",
            "7712045588",
            "January 22, 2024",
            "41.00",
            "January 31, 2025",
        ),
    )
    add(
        "nfq3456_orcr.pdf",
        "NFQ 3456",
        "or_cr",
        "en",
        or_cr(
            V["NFQ 3456"],
            "1101-00000904671",
            "6630217719",
            "June 10, 2024",
            "88.00",
            "June 30, 2025",
        ),
    )
    add(
        "car7890_orcr.pdf",
        "CAR 7890",
        "or_cr",
        "en",
        or_cr(
            V["CAR 7890"],
            "0401-00000215598",
            "5508812234",
            "October 5, 2023",
            "56.00",
            "October 31, 2024",
        ),
    )

    add(
        "nbc1234_deed_of_sale.pdf",
        "NBC 1234",
        "deed_of_sale",
        "en",
        deed_of_sale_en(
            V["NBC 1234"],
            "Maria L. Santos",
            "301 Katipunan Ave., Quezon City",
            "485,000.00",
            "FOUR HUNDRED EIGHTY-FIVE THOUSAND PESOS",
            "September 18, 2024",
            "Quezon City",
        ),
    )
    add(
        "dae9012_kasulatan_bilihan.pdf",
        "DAE 9012",
        "deed_of_sale",
        "taglish",
        deed_of_sale_taglish(
            V["DAE 9012"],
            "Carlos M. Bautista",
            "14 Gorordo Ave., Cebu City",
            "395,000.00",
            "July 6, 2024",
            "Cebu City",
        ),
    )
    add(
        "car7890_deed_of_sale.pdf",
        "CAR 7890",
        "deed_of_sale",
        "en",
        deed_of_sale_en(
            V["CAR 7890"],
            "Danilo F. Garcia",
            "9 Rizal St., Lipa City, Batangas",
            "720,000.00",
            "SEVEN HUNDRED TWENTY THOUSAND PESOS",
            "May 27, 2024",
            "Batangas City",
        ),
    )

    add(
        "nbc1234_insurance.pdf",
        "NBC 1234",
        "insurance",
        "en",
        insurance(
            V["NBC 1234"],
            "SMI-MC-2024-0031875",
            "April 1, 2024",
            "April 1, 2025",
            "520,000.00",
            "14,872.50",
            "2,000.00",
            aon=False,
            mortgagee=None,
        ),
    )
    add(
        "abc5678_insurance.pdf",
        "ABC 5678",
        "insurance",
        "en",
        insurance(
            V["ABC 5678"],
            "SMI-MC-2024-0057210",
            "August 15, 2024",
            "August 15, 2025",
            "780,000.00",
            "23,416.00",
            "3,000.00",
            aon=True,
            mortgagee=None,
        ),
    )
    add(
        "nfq3456_insurance.pdf",
        "NFQ 3456",
        "insurance",
        "en",
        insurance(
            V["NFQ 3456"],
            "SMI-MC-2024-0049923",
            "June 20, 2024",
            "June 20, 2025",
            "1,650,000.00",
            "52,305.75",
            "5,000.00",
            aon=True,
            mortgagee=BANK,
        ),
    )

    add(
        "abc5678_inspection.pdf",
        "ABC 5678",
        "inspection",
        "taglish",
        inspection_taglish(
            V["ABC 5678"],
            "KAI-2024-1187",
            "September 3, 2024",
            "48,215",
            "Engr. Rodel P. Santos",
            [
                "Hindi gumagana ang kaliwang fog lamp, kailangang palitan ang bulb.",
                "Ang apat na gulong ay 2019 pa ang manufacture date, medyo pudpod na. "
                "Estimate ng palit: PHP 18,000.",
                "May maliit na gasgas sa rear bumper, pang-touch up lang.",
                "Kumpleto ang service records sa casa hanggang 40,000 km PMS.",
            ],
            "8 / 10",
            "Pwede nang ibenta pagkatapos palitan ang gulong at fog lamp bulb.",
        ),
    )
    add(
        "nfq3456_inspection.pdf",
        "NFQ 3456",
        "inspection",
        "en",
        inspection_en(
            V["NFQ 3456"],
            "KAI-2024-0954",
            "June 12, 2024",
            "36,780",
            "Engr. Mylene A. Cortez",
            [
                "Front brake pads at about 30% remaining; replace within 5,000 km.",
                "Small dent on the right rear door, about 5 cm, no paint crack.",
                "Battery tested at 68% health; consider replacement.",
                "All recalls completed per dealer records.",
            ],
            "8.5 / 10",
            "Good buy. Replace front brake pads before resale.",
        ),
    )
    add(
        "pqr2468_inspection.pdf",
        "PQR 2468",
        "inspection",
        "taglish",
        inspection_taglish(
            V["PQR 2468"],
            "KAI-2024-1302",
            "October 1, 2024",
            "21,440",
            "Engr. Rodel P. Santos",
            [
                "Mahina ang aircon sa likod (rear AC), kailangang i-check ang blower motor.",
                "Medyo kalbo na ang dalawang gulong sa harap. Estimate: PHP 9,500.",
                "Walang palatandaan ng baha o aksidente.",
                "Ang right side mirror ay may crack sa casing.",
            ],
            "7.5 / 10",
            "Ayusin muna ang rear AC at palitan ang front tires bago i-list.",
        ),
    )

    add(
        "nfq3456_auto_loan.pdf",
        "NFQ 3456",
        "financing",
        "en",
        financing_en(
            V["NFQ 3456"],
            "BPSB-AL-2021-77351",
            "1,798,000.00",
            "359,600.00",
            "1,438,400.00",
            "6.5%",
            60,
            "31,764.67",
            "August 5, 2021",
            "July 5, 2021",
        ),
    )
    add(
        "car7890_loan_balance.pdf",
        "CAR 7890",
        "financing",
        "taglish",
        financing_taglish(
            V["CAR 7890"],
            "BPSB-AL-2019-40218",
            "950,000.00",
            "112,480.00",
            "21,086.00",
            47,
            60,
            "May 20, 2024",
        ),
    )

    # Photographed and scanned papers (OCR). "render" says how each file is produced.
    almera_orcr = or_cr(
        V["TAV 4521"],
        "0501-00000613378",
        "9934017265",
        "February 9, 2024",
        "41.00",
        "February 28, 2025",
    )
    d.append(
        dict(
            file="tav4521_cr_photo.jpg",
            plate="TAV 4521",
            doc_type="or_cr",
            language="en",
            pages=[almera_orcr[0]],
            render=dict(kind="photo", tilt=-5.5, shadow=0.55, contrast=0.85, noise=7),
        )
    )
    d.append(
        dict(
            file="tav4521_or_photo.jpg",
            plate="TAV 4521",
            doc_type="or_cr",
            language="en",
            pages=[almera_orcr[1]],
            render=dict(
                kind="photo",
                tilt=2.0,
                sideways=True,
                shadow=0.8,
                contrast=0.55,
                brightness=0.8,
                noise=6,
            ),
        )
    )
    d.append(
        dict(
            file="tav4521_deed_scan.pdf",
            plate="TAV 4521",
            doc_type="deed_of_sale",
            language="en",
            pages=deed_of_sale_en(
                V["TAV 4521"],
                "Antonio R. Medina",
                "Blk 4 Lot 9, Camella Homes, Naga City",
                "610,000.00",
                "SIX HUNDRED TEN THOUSAND PESOS",
                "August 30, 2024",
                "Naga City",
            ),
            render=dict(kind="scan", skew=1.4, noise=0.012),
        )
    )
    return d


# ---------------------------------------------------------------- Rendering


def render_pdf(path: Path, pages: list[tuple[str, str]]) -> None:
    pdf = pymupdf.open()
    for title, body in pages:
        page = pdf.new_page(width=595, height=842)  # A4
        page.insert_textbox(pymupdf.Rect(50, 45, 545, 75), title, fontsize=13, fontname="hebo")
        overflow = page.insert_textbox(
            pymupdf.Rect(50, 85, 545, 790), body, fontsize=9.5, fontname="helv"
        )
        if overflow < 0:
            raise ValueError(f"{path.name}: text overflows page '{title}'")
        page.insert_textbox(
            pymupdf.Rect(50, 805, 545, 825), FOOTER, fontsize=7, fontname="helv", align=1
        )
    pdf.set_metadata({"title": pages[0][0], "producer": "make_sample_docs.py (fictional data)"})
    pdf.save(path)
    pdf.close()


def _page_images(pages: list[tuple[str, str]], dpi: int) -> list[Image.Image]:
    """Render pages with the normal layout, then rasterize them."""
    buf = io.BytesIO()
    tmp = OUT_DIR / ".render_tmp.pdf"
    render_pdf(tmp, pages)
    with pymupdf.open(tmp) as pdf:
        images = [
            Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            for pix in (page.get_pixmap(dpi=dpi) for page in pdf)
        ]
    tmp.unlink()
    del buf
    return images


def photograph(
    page: Image.Image,
    *,
    tilt: float,
    shadow: float,
    contrast: float,
    noise: float,
    sideways: bool = False,
    brightness: float = 1.0,
    seed: int = 7,
) -> Image.Image:
    """Make a clean page look like a phone photo of paper lying on a desk."""
    rng = np.random.default_rng(seed)
    page = ImageEnhance.Contrast(page).enhance(contrast)
    page = ImageEnhance.Brightness(page).enhance(brightness)
    margin = int(page.width * 0.08)
    desk = Image.new("RGB", (page.width + 2 * margin, page.height + 2 * margin), (96, 78, 60))
    desk.paste(page, (margin, margin))
    photo = desk.rotate(tilt, resample=Image.BICUBIC, expand=True, fillcolor=(96, 78, 60))
    arr = np.asarray(photo).astype(np.float32)
    # Uneven light: brighter on one side, like a window or a phone shadow.
    gradient = np.linspace(1.0, shadow, arr.shape[1], dtype=np.float32)[None, :, None]
    arr = arr * gradient + rng.normal(0, noise, arr.shape)
    photo = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    photo = photo.filter(ImageFilter.GaussianBlur(0.8))
    if sideways:  # phone held sideways, no EXIF rotation tag
        photo = photo.rotate(90, expand=True)
    return photo


def scan(page: Image.Image, *, skew: float, noise: float, seed: int = 11) -> Image.Image:
    """Make a page look like a cheap office scan: grayscale, skewed, speckled."""
    rng = np.random.default_rng(seed)
    gray = page.convert("L").rotate(skew, resample=Image.BICUBIC, expand=False, fillcolor=255)
    arr = np.asarray(gray).copy()
    speckles = rng.random(arr.shape) < noise
    arr[speckles] = rng.integers(0, 120, speckles.sum())
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.6))


def render_document(doc: dict) -> None:
    render = doc.get("render") or {"kind": "pdf"}
    path = OUT_DIR / doc["file"]
    if render["kind"] == "pdf":
        render_pdf(path, doc["pages"])
    elif render["kind"] == "photo":
        options = {k: v for k, v in render.items() if k != "kind"}
        photo = photograph(_page_images(doc["pages"], dpi=200)[0], **options)
        photo.save(path, format="JPEG", quality=72)
    elif render["kind"] == "scan":  # image-only PDF, no text layer
        pdf = pymupdf.open()
        for image in _page_images(doc["pages"], dpi=200):
            page_image = scan(image, skew=render["skew"], noise=render["noise"])
            png = io.BytesIO()
            page_image.save(png, format="PNG", optimize=True)
            page = pdf.new_page(width=595, height=842)
            page.insert_image(page.rect, stream=png.getvalue())
        pdf.save(path, garbage=3, deflate=True)
        pdf.close()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    documents = build_documents()
    for doc in documents:
        render_document(doc)

    vehicle_fields = ("plate", "vin", "make", "model", "year", "status")
    manifest = {
        "note": "All data is fictional. Generated by scripts/make_sample_docs.py.",
        "vehicles": [{k: v[k] for k in vehicle_fields} for v in VEHICLES],
        "documents": [
            {k: doc[k] for k in ("file", "plate", "doc_type", "language")} for doc in documents
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(documents)} documents for {len(VEHICLES)} vehicles to {OUT_DIR}")


if __name__ == "__main__":
    main()
