"""Скрипт автоматического тестирования ответов ассистента.

Отправляет вопросы на /process_text_test, сохраняет ответы
рядом с эталонными для ручной проверки качества.
"""

import asyncio
import time
from typing import List, Tuple

import httpx

from client.main import wait_for_server

TEST_CASES: List[Tuple[str, str]] = [
    (
        "Какие соединения производятся микробами и могут служить прямым индикатором"
        " свежести мяса при хранении?",
        "Микробы производят летучие сульфиды. Согласно источнику: Volatile compounds"
        " produced by microbial growth such as volatile sulfides could directly"
        " indicate the freshness of meat during distribution and storage."
        " файл: Colorimetric Sensor Based on Ag-Fe NTs for H2S Sensing.pdf.",
    ),
    (
        "Что производят микроорганизмы порчи помимо изменения цвета и образования"
        " слизи?",
        "Микроорганизмы порчи производят неприятные запахи и привкусы. Согласно"
        " источнику: Spoilage microorganism is one of the main reasons for food loss"
        " and waste, which commonly produce off-odors, off-flavors, discoloration,"
        " and slime."
        " файл: Colorimetric Sensor Based on Ag-Fe NTs for H2S Sensing.pdf.",
    ),
    (
        "Концентрация каких конкретных биогенных аминов увеличивается в течение"
        " первых пяти дней порчи?",
        "В течение пяти дней возрастает концентрация кадаверина, гистамина,"
        " путресцина и тирамина. Согласно источнику: Cadaverine (CAD), histamine"
        " (HIST), putrescine (PUT), and tyramine (TYR) were detected in increasing"
        " concentrations within 5 days"
        " файл: Andre R.S. et al. Recent progress in amine gas sensors for food"
        " quality monitoring Novel architectures for sensing materials and"
        " systems.pdf.",
    ),
    (
        "Какая бактерия и какой фермент используются в качестве основы для"
        " биосенсора на путресцин?",
        "Для биосенсора используется путресциноксидаза из бактерии Kocuria rosea."
        " Согласно источнику: Putrescine biosensor based on putrescine oxidase from"
        " Kocuria rosea"
        " файл: Development of peptide impregnated VFe bimetal Prussian blue"
        " analogue as Robust nanozyme.pdf.",
    ),
    (
        "Благодаря какому именно соединению экстракт чеснока способен подавлять"
        " рост грамотрицательных и грамположительных бактерий?",
        "Экстракт чеснока подавляет бактерии благодаря антимикробному соединению"
        " аллицину. Согласно источнику: inhibit the growth of gram-negative and"
        " gram-positive bacteria due to the presence of antimicrobial compounds in"
        " the form of allicin."
        " файл: Dirpan A., Hidayat S.H. Quality and Shelf-Life Evaluation of Fresh"
        " Beef Stored in Smart Packaging.pdf.",
    ),
    (
        "Какова цепочка влияния углекислого газа на кислотность среды при упаковке"
        " пищевых продуктов?",
        "Воздействие углекислого газа приводит к резкому снижению уровня pH."
        " Согласно источнику: Upon CO2 exposure the pH dramatically decreases"
        " файл: Carbon dioxide colorimetric indicators for food packaging"
        " application. Applicability of anthocyanin and poly-lysine mixtures.pdf.",
    ),
    (
        "С какой целью углекислый газ применяется в технологиях защитной атмосферы?",
        "Углекислый газ применяется в качестве активного газа для подавления"
        " микробного метаболизма. Согласно источнику: CO2 is used as active gas in"
        " protective atmosphere technology to inhibit the microbial metabolism"
        " файл: Carbon dioxide colorimetric indicators for food packaging"
        " application. Applicability of anthocyanin and poly-lysine mixtures.pdf.",
    ),
    (
        "В каком диапазоне pH доминирует катион флавилия и какой визуальный цвет он"
        " придает?",
        "Катион флавилия доминирует в кислой среде при pH от 1 до 3 и придает яркий"
        " красный цвет. Согласно источнику: At acidic pH (range 1-3), the"
        " predominant form is the flavylium cation"
        " файл: Carbon dioxide colorimetric indicators for food packaging"
        " application. Applicability of anthocyanin and poly-lysine mixtures.pdf.",
    ),
    (
        "Какие внешние факторы окружающей среды ограничивают химическую стабильность"
        " куркумина?",
        "Стабильность куркумина ограничивается воздействием высоких температур,"
        " света и кислорода. Согласно источнику: their limited stability when exposed"
        " to relatively high temperatures (even around 60 degrees C) and other"
        " external factors (e.g., light, oxygen)"
        " файл: Cvek M. et al. Biodegradable films of PLA PPC and curcumin as"
        " packaging materials and smart indicators of food spoilage.pdf.",
    ),
    (
        "Каким образом пленки с добавлением металлоорганического каркаса ZIF67"
        " убивают клетки бактерий кишечной палочки и стафилококка?",
        "Пленки выделяют ионы кобальта, которые разрушают клеточную мембрану и"
        " вызывают гибель бактерий. Согласно источнику: the released cobalt ions"
        " possess the ability to trigger the serious damage of cell membrane and thus"
        " the death of bacteria"
        " файл: Developing strong and tough cellulose acetateZIF67 intelligent active"
        " films for shrimp freshness moni.pdf.",
    ),
    (
        "Перечислите микроорганизмы, которые способны синтезировать астаксантин.",
        "Астаксантин могут производить Haematococcus pluvialis, Phaffia rhodozyma и"
        " Chlorella vulgaris. Согласно источнику: Astaxanthin, for instance, can be"
        " produced by Haematococcus pluviali, Phaffia rhodozyma, and Chlorella"
        " vulgaris."
        " файл: Yu Z. et al. Boosting Food System Sustainability through Intelligent"
        " Packaging Application of Biodegradable Freshness Indicators.pdf.",
    ),
    (
        "На какие молекулярные компоненты фермент лактаза расщепляет лактозу в"
        " организме?",
        "Лактаза расщепляет лактозу на глюкозу и галактозу. Согласно источнику:"
        " Lactase is an enzyme that breaks down the disaccharide lactose to its"
        " component parts, glucose and galactose."
        " файл: squad:570cf364fed7b91900d45b53.",
    ),
    (
        "Какова ферментативная цепочка преобразования сахара в уксусную кислоту при"
        " созревании гуавы?",
        "Сахар ферментативно превращается в этанол, который окисляется в"
        " ацетальдегид, а затем в уксусную кислоту. Согласно источнику: this sugar"
        " was converted enzymatically to ethanol, oxidized into acetaldehyde, which"
        " is then further oxidized into acetic acid"
        " файл: Real time on-package freshness indicator for guavas packaging.pdf.",
    ),
    (
        "Какие две молекулы образуются из энергии света, захваченной хлорофиллом А в"
        " процессе фотосинтеза?",
        "Энергия используется для создания молекул АТФ и НАДФН. Согласно источнику:"
        " The light energy captured by chlorophyll a is initially in the form of"
        " electrons (and later a proton gradient) that's used to make molecules of"
        " ATP and NADPH"
        " файл: squad:5726be0b5951b619008f7ccf.",
    ),
    (
        "Каков механизм действия антибиотиков группы бета-лактамов, к которым"
        " относится пенициллин?",
        "Пенициллин подавляет образование поперечных связей пептидогликана в"
        " клеточной стенке бактерий. Согласно источнику: beta-Lactam antibiotics,"
        " such as penicillin, inhibit the formation of peptidoglycan cross-links in"
        " the bacterial cell wall."
        " файл: squad:572fb096a23a5019007fc8a1.",
    ),
    (
        "Против каких конкретно бактерий пигмент шиконин проявляет наибольшую"
        " противомикробную активность?",
        "Шиконин особенно эффективен против грамположительных бактерий, поскольку он"
        " разрушает целостность их клеточных мембран. Согласно источнику: SKN has"
        " antimicrobial ability, especially against Gram-positive bacteria, by"
        " disrupting the integrity of microbial cell membranes"
        " файл: Dual-functional shikonin-loaded quaternized"
        " chitosanpolycaprolactone nanofibrous film with pH-sensing for active and"
        " intelligent food packaging.pdf.",
    ),
    (
        "С какой целью применяется метод хранения в условиях замораживания-охлаждения"
        " в комбинации с полифенолами чая?",
        "Этот метод используется для контроля меланоза, предотвращения ухудшения"
        " качества и подавления роста бактерий порчи. Согласно источнику: Efficacy of"
        " freeze-chilled storage combined with tea polyphenol for controlling"
        " melanosis, quality deterioration, and spoilage bacterial growth"
        " файл: A fast-response visual indicator film based on polyvinyl"
        " alcohol.pdf.",
    ),
    (
        "На какие вещества распадаются бетацианины в щелочной среде, приобретая"
        " желтую окраску?",
        "В щелочной среде бетацианины распадаются на бесцветный цикло-ДОФА"
        " 5-О-глюкозид и желтую беталаминовую кислоту. Согласно источнику:"
        " betacyanins degrade into colorless cyclo-DOPA"
        " 5-O-(malonyl)-beta-glucosides and yellow betalamic acids at an alkaline"
        " condition."
        " файл: Yu Z. et al. Boosting Food System Sustainability through Intelligent"
        " Packaging Application of Biodegradable Freshness Indicators.pdf.",
    ),
    (
        "Какое вещество молочнокислые бактерии используют в качестве источника"
        " энергии в пищевых продуктах?",
        "Молочнокислые бактерии потребляют глюкозу. Согласно источнику: lactic acid"
        " bacteria consume glucose in foods as an energy source"
        " файл: Yu Z. et al. Boosting Food System Sustainability through Intelligent"
        " Packaging Application of Biodegradable Freshness Indicators.pdf.",
    ),
    (
        "Какие химические реакции происходят с аммиаком, выделяемым при порче"
        " продуктов, при его контакте с водной средой?",
        "Аммиак поглощается водной средой и распадается на ионы аммония и"
        " гидроксид-ионы, формируя щелочную среду. Согласно источнику: The NH3"
        " produced after food spoilage decomposes into NH4+ and OH- after absorption"
        " by aqueous media to build an alkaline environment"
        " файл: A fast-response visual indicator film based on polyvinyl"
        " alcohol.pdf.",
    ),
    (
        "Что происходит с содержанием амилозы при нагревании овсяного крахмала в"
        " субкритической воде до 115 градусов?",
        "Содержание амилозы увеличивается до 30 процентов из-за ее высвобождения из"
        " спиральной структуры крахмала. Согласно источнику: The amylose content of"
        " NOS was recorded as 26.58% and further increased to 30.12% when subjected"
        " to subcritical water treatment up to some extent (115 degrees C)."
        " файл: Smart packaging film prepared from subcritical water-modified oat"
        " starch and betalain of beetroot extract reinforced with cellulose"
        " nanofibrils.pdf.",
    ),
    (
        "Какой гриб в естественных условиях синтезирует молекулу мевастатина?",
        "Мевастатин синтезируется грибом Penicillium citrinum. Согласно источнику:"
        " identified mevastatin (ML-236B), a molecule produced by the fungus"
        " Penicillium citrinum"
        " файл: squad:571ae94232177014007e9fda.",
    ),
    (
        "Как повышенная температура воздуха влияет на микробиологическое качество"
        " свежего мяса?",
        "Высокие температуры воздуха вызывают ускоренное заражение мяса патогенными"
        " микроорганизмами. Согласно источнику: high air temperatures, consequently"
        " accelerated pathogenic microbial contamination"
        " файл: Dirpan A., Hidayat S.H. Quality and Shelf-Life Evaluation of Fresh"
        " Beef Stored in Smart Packaging.pdf.",
    ),
    (
        "Как температура и влажность влияют на стабильность цвета красителей на"
        " основе антоцианов?",
        "Влага, кислород и высокая температура проникают в пленку и разрушают"
        " стабильность цвета антоцианов. Согласно источнику: allowing moisture,"
        " oxygen, and temperature to infiltrate the film and degrade anthocyanin"
        " color stability"
        " файл: Hashim S. B. H. et al. Development of Smart Colorimetric Sensing"
        " Films Carbohydrate-Based with Soybean Wax and Purple Cauliflower"
        " Anthocyanins for Visual Monitoring of Shrimp Freshness.pdf.",
    ),
    (
        "Почему плотоядные животные обязаны употреблять в пищу мясо других животных?",
        "Плотоядные животные едят мясо для получения определенных витаминов и"
        " питательных веществ, которые их организм не может синтезировать"
        " самостоятельно. Согласно источнику: obligate carnivores must eat animal"
        " meats to obtain certain vitamins or nutrients their bodies cannot otherwise"
        " synthesize"
        " файл: squad:5726fdd95951b619008f8431.",
    ),
]

API_URL = "http://localhost:8000/process_text_test"
OUTPUT_FILE = "tests/test_results.txt"


async def run_tests():
    """Запускает все тест-кейсы последовательно и сохраняет результаты в файл."""
    print(f"Запуск тестов. Всего вопросов: {len(TEST_CASES)}")
    await wait_for_server("http://localhost:8000")

    results = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for idx, (question, expected) in enumerate(TEST_CASES, 1):
                start = time.perf_counter()
                print(f"[{idx}/{len(TEST_CASES)}] Обработка вопроса: {question}")

                try:
                    response = await client.post(API_URL, json={"text": question})
                    response.raise_for_status()
                    data = response.json()
                    current_answer = data.get("answer", "ОТВЕТ НЕ ПОЛУЧЕН (пусто)")
                except Exception as e:
                    current_answer = (
                        f"ОШИБКА ПРИ ВЫПОЛНЕНИИ ЗАПРОСА: {type(e).__name__}: {e}"
                    )

                elapsed = time.perf_counter() - start
                report_chunk = (
                    f"ВОПРОС {idx}:\n{question}\n\n"
                    f"ОТВЕТ ТЕКУЩИЙ:\n{current_answer}\n\n"
                    f"ПРАВИЛЬНЫЙ ОТВЕТ:\n{expected}\n\n"
                    f"ВРЕМЯ ОТВЕТА:\n{elapsed:.2f} сек\n\n"
                    f"{'-' * 40}\n\n"
                )
                print(f"Время ответа: {elapsed:.2f}s")
                results.append(report_chunk)
    finally:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.writelines(results)

    print(f"Тестирование завершено. Результаты сохранены в {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except Exception as e:
        print(f"ОШИБКА ПРИ ВЫПОЛНЕНИИ ЗАПРОСА: {e}")
