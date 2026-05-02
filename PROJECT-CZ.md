# Detekce anomálií v šifrované komunikaci pomocí AI

**Bc. Aleksandra Parkhomenko**  
**Vedoucí DP:** prof. RNDr. Jiří Ivánek, CSc.  
**Semestr zpracování Projektu DP:** LS 2025/2026

---

## Úvod

V roce 2026 představuje kybernetická bezpečnost jednu z klíčových oblastí informačních technologií, a s rostoucím využíváním digitálních služeb a síťové komunikace narůstá počet i sofistikovanost kybernetických hrozeb, které ohrožují data a IT infrastrukturu organizací, a následně i jejich zisk a reputaci. Ochrana ICT systémů proto vyžaduje nejen hardwarová a softwarová opatření, ale také vhodné procesy a techniky detekce bezpečnostních incidentů, jako jsou například systémy detekce průniku (IDS) (Dutta, N., Jadav, N., Tanwar, S., Sarma, H. K. D., & Pricop, E., 2022), aby byla organizace i v případě úspěšného útoku (nebo jeho části) včas informována a začala situaci napravovat.

Podíl šifrované a nešifrované komunikace se z roku 2013 na rok 2022 změnil ze 48 % na 95 % (Desai, 2022). Tento trend je z mnoha pohledů pozitivní, ale také negativní: protokoly jako TLS (Transport Layer Security) a různé VPN (Virtual Private Network) sice chrání soukromí uživatelů a firemní data, ale zároveň komplikují tradiční metody detekce založené na analýze obsahu paketů. Útočníci této situace aktivně využívají – provádějí skryté útoky, exfiltraci dat a vytvářejí skryté komunikační kanály (Barut, O., Grohotolski, M., DiLeo, C., Luo, Y., Li, P., & Zhang, T., 2020). Studie ukazují, že v roce 2022 se 85 % útoků uskutečnilo přes šifrované kanály (Zscaler ThreatLabz, 2022). V současné situaci je proto nezbytné používat metody detekce, které nepotřebují znát obsah, ale vystačí si se základními údaji o struktuře dat a časovými údaji o dynamice provozu. Z tohoto důvodu moderní přístupy analýzy síťového provozu využívají strojové učení a umělou inteligenci k odhalování anomálií na základě metadat a statistických vlastností bez nutnosti dešifrování, čímž zachovávají soukromí uživatelů.

---

## Vymezení problému

Jak bylo zmíněno výše, rostoucí míra šifrování síťové komunikace mimo jiné způsobuje, že tradiční metody detekce kybernetických hrozeb založené na analýze obsahu paketů postupně ztrácejí účinnost. Současný výzkum ukazuje, že tradiční modely využívající hloubkovou inspekci paketů (DPI) se staly v prostředí šifrovaných komunikačních protokolů, jako je TLS, fakticky nepoužitelné – bez přístupu k obsahu přenášených dat jsou bezpečnostní nástroje výrazně omezeny v možnosti odhalit škodlivé aktivity, konkrétně mají méně informací na vstupu (Papadogiannaki, E., Tsirantonakis, G., & Ioannidis, S., 2022).

Detekce anomálií v šifrovaném provozu je otevřenou vědeckou výzvou, která vyžaduje nové přístupy využívající metody umělé inteligence a analýzu statistických charakteristik provozu místo obsahové inspekce. Výzkumy potvrzují, že klasifikace šifrovaného provozu se stala kritickou součástí moderního síťového managementu a cloudových bezpečnostních služeb, a metody strojového a hlubokého učení dokáží efektivně klasifikovat provoz i bez přístupu k obsahu paketů (Almuhammadi, S., Alnahari, N., & Ba-amer, R., 2025).

Problém spočívá v nalezení efektivních metod detekce anomálií v šifrované komunikaci, které umožní odhalovat hrozby bez narušení soukromí uživatelů, a zároveň v posouzení zda je využití na zdroje náročné umělé inteligence nutné a vhodné pro konkrétní případy.

Cílovou skupinou, která má z vyřešení problému užitek, jsou především provozovatelé podnikových sítí a bezpečnostní týmy, například SOC (Security Operations Center) operátoři, kteří potřebují efektivně monitorovat provoz bez dešifrování komunikace, a koncoví uživatelé, jejichž komunikace zůstává šifrovaná, a proto lépe chráněná.

---

## Rešeršní strategie rešerše provedené v Projektu DP

V rámci zpracování Projektu DP byla provedena prvotní rešerše odborných zdrojů zaměřená na detekci anomálií v šifrované síťové komunikaci, využití metod umělé inteligence v kybernetické bezpečnosti a analýzu síťových toků.

### Rešeršní otázky

1. Jaké typy útoků existují a jaké jsou aktuální trendy?
2. Jaké metody strojového učení se používají k detekci anomálií v šifrované síti?
3. Jaké klasické (statistické) metody se používají k detekci anomálií v šifrované síti?

### Vyhledávací řetězce

Vyhledávací řetězce zahrnovaly odpovídající otázkám kombinace:

- **(TLS OR HTTPS OR SSL) AND ("intrusion detection" OR "threat detection" OR "anomaly") AND (malware OR attack OR exfiltration OR lateral)**: aktuální typy útoků a trendy
- **("encrypted traffic" OR "encrypted communication" OR "TLS traffic") AND ("anomaly detection" OR "intrusion detection") AND ("machine learning" OR "deep learning" OR "neural network")**: články a práce o využití AI/ML
- **("encrypted traffic" OR "network traffic") AND ("anomaly detection" OR "intrusion detection") AND ("statistical" OR "rule-based") -"AI" -"deep learning"**: články o klasických a statistických metodách

### Databáze

Rešerše byla prováděna primárně v následujících databázích:

- IEEE Xplore
- Theses.cz
- SpringerLink
- ACM Digital Library
- Google Scholar, který agreguje výsledky z výše uvedených a dalších databází

### Filtry a kritéria odfiltrování

- Omezení na publikace od roku 2020 pro zajištění aktuálnosti zdrojů
- Preferována odborná recenzovaná literatura, články z konferencí a renomovaných časopisů, akademické diplomové práce a disertace
- Vzhledem k vysoké popularitě tématu byly "nepopulární" články pokud možno odfiltrovány podle počtu citací
- Zúžení výsledků bylo provedeno kontrolou názvů a abstraktů dle relevantnosti k tématu

### Výsledky rešerše – aktuální typy útoků a trendy

| Databáze | Prvotní počet výsledků | Počet výsledků po filtrování (rok ≥2020) | Počet relevantních výsledků, vybraných pro rešerši |
|----------|------------------------|-------------------------------------------|-----------------------------------------------------|
| Google Scholar | 876 000 | 36 200 | 1 |
| SpringerLink | 1 510 | 227 | 2 |
| IEEE Xplore | 6 | 5 | 3 |
| Theses.cz | 1 | 1 | - |

**Tabulka 1.** Výsledky rešerše o aktuálních typech útoků a trendech v šifrované síťové komunikaci podle databáze. Mezi relevantní výsledky byly zahrnuty pouze zdroje použité pro formulaci kontextu v kryptografickém světě.

Rešeršní otázka pro tuto část rešerše byla stanovena velmi široce a mezi "relevantní" výsledky v tabulce 1 byly zahrnuty pouze zdroje použité pro formulaci kontextu v kryptografickém světě. Například byl nalezen článek poskytující přehled současného stavu detekce malwaru v TLS šifrovaném provozu (Keshkeh, K., Jantan, A., Alieyan, K., & Gana, U. M., 2021).

### Výsledky rešerše – Detekce anomálií v šifrované síti pomocí AI/ML

| Databáze | Prvotní počet výsledků | Počet výsledků po filtrování (rok ≥2020) | Počet relevantních výsledků, vybraných pro rešerši |
|----------|------------------------|-------------------------------------------|-----------------------------------------------------|
| Google Scholar | 10 900 | 8 480 | 7 |
| SpringerLink | 848 | 698 | 3 |
| IEEE Xplore | 187 | 17 | 1 |
| Theses.cz | 4 | 3 | 1 |

**Tabulka 2.** Výsledky systematického vyhledávání zdrojů o využití AI/ML pro detekci anomálií v šifrované síťové komunikaci.

Vybrané téma se ukázalo být velmi populární zejména v posledních dvou letech, ve kterých byla vydána nadpoloviční část prací zahrnutých v tabulce 2. Bylo nalezeno několik vědeckých článků s podobným zaměřením, například článek věnovaný ochraně soukromí pomocí strojového učení pro klasifikaci šifrovaného provozu v bezpečných cloudových službách (Almuhammadi, S., Alnahari, N., & Ba-amer, R., 2025). Taktéž byl nalezen například systematický přehled literatury technologie detekce anomálií na základě umělé inteligence ve šifrovaném provozu (Ji, I. H., Lee, J. H., Kang, M. J., Park, W. J., Jeon, S. H., & Seo, J. T., 2024), což poslouží podporou rešerše.

### Výsledky rešerše – Klasické a statistické metody detekce anomálií v šifrované síti

| Databáze | Prvotní počet výsledků | Počet výsledků po filtrování (rok ≥2020) | Počet relevantních výsledků, vybraných pro rešerši |
|----------|------------------------|-------------------------------------------|-----------------------------------------------------|
| Google Scholar | 15 800 | 3 390 | 2 |
| SpringerLink | 1 010 | 283 | 2 |
| IEEE Xplore | 729 | 43 | - |
| Theses.cz | 10 | 8 | 1 |

**Tabulka 3.** Výsledky systematického vyhledávání literatury o klasických a statistických metodách pro detekci anomálií v šifrované síťové komunikaci.

Pro lepší pochopení alternativních přístupů byly vyhledány články o klasických ne-AI metodách. Tady je hranice mezi umělou inteligencí a tradičním strojovým učením skutečně vágní, proto pro účely této části rešerše bylo definováno, že AI zahrnuje pokročilé techniky jako neuronové sítě a hluboké učení, zatímco se hledání zaměřilo na ne-AI statistické a heuristické metody, vizte tabulku 3. Například se ukázalo, že efektivními příklady podle některých článků jsou PCA (Principal Component Analysis) a K-means clustering (Chapagain, P., Timalsina, A., Bhandari, M., & Chitrakar, R., 2022).

Celkem bylo nalezeno 23 zdrojů, které poslouží osnovou teoretické části diplomové práce.

---

## Cíle diplomové práce

Hlavním cílem diplomové práce je vyhodnotit vhodnost použití metod umělé inteligence pro detekci anomálií v šifrované síťové komunikaci prostřednictvím systematického srovnání AI a tradičních metod v experimentálním prostředí. 

### Dílčí cíle

1. Identifikovat a analyzovat aktuální metody umělé inteligence a tradiční statistické metody vhodné pro detekci anomálií v šifrovaném provozu na základě rešerše literatury.
2. Navrhnout experimentální prostředí umožňující sběr, generování a validaci dat síťových toků pro testování detekčních metod.
3. Implementovat prototyp detekčního systému využívajícího vybrané AI algoritmy i tradiční metody.
4. Vyhodnotit účinnost implementovaných přístupů na připravených datech pomocí standardních metrik: přesnost, úplnost (která je v tomto kontextu velmi důležitá), F1 skóre a ROC křivka.
5. Systematicky porovnat efektivitu metod AI a tradičních přístupů a identifikovat situace, ve kterých je použití AI přínosné.

---

## Výzkumné otázky a hypotézy

Výzkumné otázky pro tuto diplomovou práci jsou formulovány následovně:

1. Jaké jsou způsoby detekce anomálií v šifrovaném provozu bez dešifrování (pouze z metadat)?
2. Jakou účinnost mají metody umělé inteligence oproti tradičním statistickým metodám při detekci anomálií v šifrovaném síťovém provozu?
3. Jaké požadavky musí splňovat návrh experimentálního prostředí pro validaci metod detekce anomálií v šifrovaném provozu?
4. Jaká jsou omezení a výzvy při implementaci detekčních metod do reálných bezpečnostních systémů?

Součástí diplomové práce je řízený experiment srovnávající AI a tradiční metody na datasetech s měřením standardních metrik, proto je formulována také následující hypotéza:

**Hypotéza:** Metody hlubokého učení dosahují vyšší hodnoty F1 skóre než tradiční statistické metody při detekci anomálií v šifrovaném provozu.

---

## Metody použité k dosažení cílů

Pro dosažení hlavního cíle diplomové práce, tj. vyhodnocení vhodnosti použití metod umělé inteligence pro detekci anomálií v šifrované síťové komunikaci, bude využita výzkumná strategie **Design Science Research** (Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S., 2007), která je vhodná pro práce zaměřené na návrh a vyhodnocení artefaktů v oblasti informačních technologií. Tato strategie poskytuje rámec pro systematický návrh, implementaci a validaci detekčních metod v experimentálním prostředí.

Konkrétní metody jsou mapovány na dílčí cíle následovně:

1. Na základě **systematické rešerše literatury** budou identifikovány a analyzovány metody umělé inteligence (například neuronové sítě, hluboké učení, rekurentní sítě) a tradiční statistické metody vhodné pro detekci anomálií v šifrovaném síťovém provozu. Výsledky rešerše budou shrnuty pomocí **tematické analýzy** (Braun, V., & Clarke, V., 2006), která umožní kategorizaci přístupů podle typu metody, oblasti aplikace a dosažených výsledků.

2. Bude navrženo **experimentální prostředí** pro testování detekčních metod. Návrh bude zahrnovat definici architektury pro sběr a reprodukovatelné přehrávání síťových toků, konfiguraci scénářů provozu (běžný provoz, útoky, anomální chování) a výběr nástrojů pro zachytávání metadat síťových toků. Pro experimenty budou primárně využity veřejně dostupné datasety šifrovaného provozu (například CICIDS, UNSW-NB15), případně doplněné synteticky generovanými daty podle potřeby.

3. Bude **implementován softwarový prototyp** detekčního systému využívající vybrané AI algoritmy i tradiční metody. Implementace bude zahrnovat předzpracování dat (čištění, normalizaci, výběr a extrakci příznaků z metadat síťových toků), trénování modelů a jejich integraci do testovacího prostředí.

4. Výkonnost implementovaných přístupů bude vyhodnocena pomocí **standardních metrik** používaných v oblasti detekce anomálií: přesnost (precision), úplnost (recall), F1 skóre a ROC křivka. Vyhodnocení bude provedeno pomocí metod **deskriptivní statistiky** pro sumarizaci výsledků a **inferenční statistiky** pro testování statistické významnosti rozdílů mezi metodami.

5. Výsledky AI a tradičních metod budou systematicky porovnány na shodných datasetech a scénářích pomocí **experimentu** (Wohlin, C., Runeson, P., Höst, M., Ohlsson, M. C., Regnell, B., & Wesslén, A., 2012), který umožní kontrolované srovnání přístupů a identifikaci situací, ve kterých je použití AI přínosné oproti tradičním metodám.

---

## Omezení diplomové práce

Hlavním omezením je **dostupnost a kvalita datasetů** pro trénování a testování modelů detekce anomálií: veřejně dostupné datasety šifrovaného provozu nemusí plně pokrývat aktuální typy útoků a různorodost reálných síťových prostředí, a synteticky generovaná data nemusí zachytit komplexitu běžného provozu (generování vhodného datasetu je velice komplexní výzvou a je samo o sobě téma pro disertační práce).

Dalším omezením je **rozsah pokrytých scénářů**, který je omezen na vybrané typy anomálií a útoků, které nelze považovat za vyčerpávající reprezentaci všech kybernetických hrozeb. Výsledky tedy nelze bez dalšího ověření zobecnit na všechny varianty útoků v šifrovaném provozu.

Součástí práce je experiment a **validita a generalizovatelnost výsledků** budou ovlivněny experimentálními podmínkami. Závěry získané v testovacím prostředí nemusí plně odpovídat chování modelů v reálných heterogenních sítích s proměnlivými charakteristikami provozu, na tento problém už narazili výzkumníci v rámci obdobných prací (Qing, Y., Yin, Q., Deng, X., Zhang, X., Li, P., Liu, Z., Sun, K., Xu, K., & Li, Q., 2025). Aplikace výsledků v různých organizacích a síťových topologiích tedy vyžaduje další validaci.

Posledním omezením je **závislost výkonnosti modelů** na kvalitě, reprezentativnosti a vyvážení trénovacích dat, což může ovlivnit schopnost detekovat méně časté typy anomálií. V práci bude proto diskutováno, za jakých podmínek jsou dosažené výsledky platné a jaká jsou doporučení pro praktickou aplikaci.

---

## Význam a přínos diplomové práce

Diplomová práce přispěje k rozvoji poznání v oblasti kybernetické bezpečnosti systematickou analýzou a srovnáním metod umělé inteligence a tradičních statistických přístupů pro detekci anomálií v šifrované síťové komunikaci. Rešerše a identifikace aktuálních metod poskytnou ucelený přehled dostupných technik a jejich vhodnosti pro analýzu šifrovaného provozu bez narušení soukromí uživatelů. Experimentální prostředí vytvořené v rámci diplomové práce umožní reprodukovatelné testování detekčních metod na datech síťových toků a může sloužit jako základ pro další výzkum.

Funkční prototyp detekčního systému implementující vybrané AI i tradiční metody může být integrován do bezpečnostních nástrojů jako IDS/IPS (Intrusion Detection/Prevention Systems), samozřejmě po speciálních úpravách a nastavení pro konkrétní podnik. Aby bezpečnostní týmy byly schopny informovaně volit vhodné technologie, bude provedeno systematické srovnání pomocí standardních metrik, zejména úplnosti, která je kritická pro minimalizaci falešně negativních detekcí, a budou identifikovány konkrétní situace, kdy je použití AI oproti tradičním metodám přínosné.

Celkově výsledky práce zvýší efektivitu detekce hrozeb v prostředí rostoucího šifrování, kde tradiční metody založené na analýze obsahu ztrácejí účinnost, poskytnou provozovatelům podnikových sítí a SOC týmům praktická doporučení pro implementaci detekčních řešení, která respektují soukromí uživatelů a současně efektivně chrání před kybernetickými útoky skrytými v šifrované komunikaci.

---

## Zdroje

Almuhammadi, S., Alnahari, N., & Ba-amer, R. (2025). Privacy-preserving machine learning for encrypted traffic classification in secure cloud services. In *Proceedings of the IEEE/ACM 12th International Conference on Big Data Computing, Applications and Technologies (BDCAT '25)* (Article 23, 6 pp.). Association for Computing Machinery. https://doi.org/10.1145/3773276.3774875

Qing, Y., Yin, Q., Deng, X., Zhang, X., Li, P., Liu, Z., Sun, K., Xu, K., & Li, Q. (2025). Training robust classifiers for classifying encrypted traffic under dynamic network conditions. In *Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security (CCS '25)* (pp. 3564–3578). Association for Computing Machinery. https://doi.org/10.1145/3719027.3765073

Papadogiannaki, E., Tsirantonakis, G., & Ioannidis, S. (2022). Network intrusion detection in encrypted traffic. *2022 IEEE Conference on Dependable and Secure Computing (DSC)*, 1–8. https://doi.org/10.1109/DSC54232.2022.9888942

Ji, I. H., Lee, J. H., Kang, M. J., Park, W. J., Jeon, S. H., & Seo, J. T. (2024). Artificial Intelligence-Based Anomaly Detection Technology over Encrypted Traffic: A Systematic Literature Review. *Sensors, 24*(3), 898. https://doi.org/10.3390/s24030898

Barut, O., Grohotolski, M., DiLeo, C., Luo, Y., Li, P., & Zhang, T. (2020). Machine learning based malware detection on encrypted traffic: A comprehensive performance study. In *Proceedings of the 7th International Conference on Networking, Systems and Security (NSysS '20)* (pp. 45–55). Association for Computing Machinery. https://doi.org/10.1145/3428363.3428365

Chapagain, P., Timalsina, A., Bhandari, M., & Chitrakar, R. (2022). Intrusion detection based on PCA with improved K-means. In S. Mekhilef, R. N. Shaw, & P. Siano (Eds.), *Innovations in electrical and electronic engineering. ICEEE 2022* (Lecture Notes in Electrical Engineering, Vol. 894). Springer. https://doi.org/10.1007/978-981-19-1677-9_2

Keshkeh, K., Jantan, A., Alieyan, K., & Gana, U. M. (2021). A review on TLS encryption malware detection: TLS features, machine learning usage, and future directions. In N. Abdullah, S. Manickam, & M. Anbar (Eds.), *Advances in cyber security. ACeS 2021* (Communications in Computer and Information Science, Vol. 1487). Springer. https://doi.org/10.1007/978-981-16-8059-5_13

Ferdous, J., Islam, R., Mahboubi, A., & Islam, M. Z. (2023). A review of state-of-the-art malware attack trends and defense mechanisms. *IEEE Access, 11*, 121118–121141. https://doi.org/10.1109/ACCESS.2023.3328351

Dutta, N., Jadav, N., Tanwar, S., Sarma, H. K. D., & Pricop, E. (2022). Intrusion detection systems fundamentals. In *Cyber security: Issues and current trends* (Studies in Computational Intelligence, Vol. 995). Springer. https://doi.org/10.1007/978-981-16-6597-4_6

Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. *Qualitative Research in Psychology, 3*(2), 77–101.

Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. *Journal of Management Information Systems, 24*(3), 45–77.

Wohlin, C., Runeson, P., Höst, M., Ohlsson, M. C., Regnell, B., & Wesslén, A. (2012). *Experimentation in software engineering*. Springer.

Desai, D. (2022, December 22). Encrypted traffic, once thought safe, now responsible for most cyberthreats. *Dark Reading*. https://www.darkreading.com/application-security/encrypted-traffic-once-thought-safe-now-responsible-for-most-cyberthreats

Zscaler ThreatLabz. (2022). *The state of encrypted attacks, 2022*. Zscaler. https://info.zscaler.com/resources-industry-reports-the-state-of-encrypted-attacks-2022
