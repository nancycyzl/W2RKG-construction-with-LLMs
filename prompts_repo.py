def get_few_shot_examples(style, k, shot_ids=None):

    abstract_list, w2r_list = get_abstract_list()

    if shot_ids:
        # get the specified examples by id
        assert len(shot_ids) == k, "shot_ids {} must have length equal to shot_length {}".format(shot_ids, k)
        abstract_list = [abstract_list[i] for i in shot_ids]
        w2r_list = [w2r_list[i] for i in shot_ids]
    else:
        # get the first k examples
        abstract_list = abstract_list[:min(len(abstract_list), k)]
        w2r_list = w2r_list[:min(len(w2r_list), k)]

    examples_list = [wrap_example(abstract, w2r, style) for abstract, w2r in zip(abstract_list, w2r_list)]
    examples_str = "\n\n".join(examples_list)

    return examples_str


def get_system_prompt_basic(style="json"):

    if style == "json":
        system_prompt_basic = '''Given a paragraph, extract the waste-to-resource transformation in a json format that includes
            waste, transforming_process, and transformed_resource. The format should be: 
        {
            "waste": [],
            "transforming_process": [],
            "transformed_resource": []
        }
        You should only use names or phrases to describe waste, transforming_process, and transformed_resource.
        Keep the value empty if there is no corresponding information. Do not include any explanation. Do not include nested json content.
        '''
    elif style == "code":
        system_prompt_basic = '''Extract the waste-to-resource information from the given paragraph. You should extract related wastes,
        transforming processes, transformed resources as a list and then assign it to each key in the w2r dictionary in this format:
        w2r["waste"] = [waste1, waste2, ...]
        w2r["transforming_process"] = [process1, process2, ...]
        w2r["transformed_resource"] = [resource1, resource2, ...]
        Directly output the complete code. Do not use abbreviations. Do not import any library. Do not explain yourself.
        '''
    else:
        raise "Wrong style!"

    return system_prompt_basic


def wrap_example(abstract, w2r, style):
    if style == "json":
        example = '''
        Text: {}
        Result: {}
        '''.format(abstract, w2r)

    elif style == "code":
        example = '''
        w2r = {{"waste": [], "transforming_process": [], "transformed_resource": []}}
        text = "{}"
        # now complete the code by extracting waste, transforming_process, transformed_resource
        w2r["waste"]={}
        w2r["transforming_process"]={}
        w2r["transformed_resource"]={}
        '''.format(abstract, w2r["waste"], w2r["transforming_process"], w2r["transformed_resource"])

    else:
        raise ValueError("Wrong style!")

    return example


def get_abstract_list():

    e0 = '''The valorization of winery pomace, traditionally considered waste, offers a sustainable approach to 
            arnessing its bioactive compounds. In this study, we investigated the antiproliferative potential of winery pomace
            extract against thyroid cancer cells. An ultrasound-assisted extraction technique was employed to extract the
            bioactive compounds. The optimal conditions for this process were determined through a Rotational Central Composite
            Design (50 °C, 4 g/L solid-to-liquid ratio, 40% ethanol concentration, and 79 W ultrasound power). The resulting
            extract demonstrated robust antioxidant activity, reflected in the elevated total phenolic content (TPC) and free
            radical scavenging ability. Furthermore, the effect of the extract on peripheral blood mononuclear cells (PBMCs)
            and TPC-1 cancer cells was assessed. PBMCs exposed to the extract at 10 mg/mL concentration exhibited enhanced
            viability, whereas cancer cells displayed concentration-dependent cytotoxic responses, indicating a selective
            anticancer effect. These findings underscore the potential of winery pomace extract as a natural antioxidant with
            promising anticancer properties, warranting further investigation for potential applications in cancer therapy. 
            Moreover, to the best of the authors' knowledge, this work represents one of the first publications to demonstrate
            the potential of wine pomace extract to inhibit thyroid cancer cells.'''
    
    e0_w2r = {"waste": ["winery pomace"],
                    "transforming_process": ["ultrasound-assisted extraction of bioactive compounds"],
                    "transformed_resource": ["natural antioxidant"]}
    
    e1 = '''Footwear industries generate leather waste during the operation. Some of these wastes contain
            chromium, which may bring environmental concerns. This study aimed to reuse finished leather waste, the major
            part of these hazardous wastes, via producing a composite with thermoplastic polyurethane (TPU) for shoe soles.
            Finished leather waste containing black dyes and pigments was used to color the TPU. The finished leather waste
            was fragmented, milled, micronized and blended with TPU in a ratio of 10%, 15%, and 20% w/w to produce composite
            materials. The composite materials were evaluated by morphological and thermal characterizations, physical–mechanical
            analysis, and environmental tests (leaching and solubilization), which presented that the physical–mechanical and
            thermal properties were within the standard of shoe soles, and the composites can be classified as non-hazardous.
            The composites enabled a new way of coloring polymeric matrices and reusing leather waste.'''
    
    e1_w2r = {"waste": ["finished leather waste"],
                    "transforming_process": ["composing", "fragmentation", "milling", "microning", "blending"],
                    "transformed_resource": ["shoe soles"]}
    
    e2 = '''The valorization of wastes as an alternative or secondary raw material in various products and processes
            has been a solution for the implementation of sustainability, a safer environment, and the concept of circular
            economy in the efficient use and management of natural resources. To promote sustainability through a circular
            economy approach, this work tries to demonstrate the environmental gains that are obtained by bringing together,
            in an industrial symbiosis action, two large industrial sectors (the pulp and paper and the road pavement sectors)
            responsible for generating large amounts of wastes. A sustainability assessment, based on a life cycle and circular
            economy approach, is presented here, and discussed using a simple case study carried out on a real scale. Two wastes
            (dregs and grits) from the pulp and paper industry (PPI) were used to partially replace natural fine aggregates
            in the production of bituminous mixtures used on the top surface of road pavements. The impacts at a technical,
            environmental, economic, and social level were assessed and it was shown that this simple waste valorization action
            is not only positive for the final product from a technical point of view, but also for the environment, causing
            positive impacts on the different sustainability dimensions that were evaluated.'''
    
    e2_w2r = {"waste": ["pulp and paper industry dregs", "pulp and paper industry grits"],
                    "transforming_process": ["replacing natural fine aggregates"],
                    "transformed_resource": ["bituminous mixtures for road pavements"]}
    
    e3 = '''This study develops a set of measures to address the interrelationship among circular waste-based
            bioeconomy (CWBE) attributes, including those of government strategy, digital collaboration, supply chain
            integration, smart operations, and a green supply chain, to build a circular bioeconomy that feeds fish waste
            back into the economy. CWBE development is a potential solution to the problem of waste reuse in the fish
            supply chain; however, this potential remains untapped, and prior studies have failed to provide the criteria
            to guide its practices. Such an analytical framework requires qualitative assessment, which is subject to
            uncertainty due to the linguistic preferences of decision makers. Hence, this study adopts the fuzzy Delphi
            method to obtain a valid set of attributes. A fuzzy decision-making trial and evaluation was applied to address
            the attribute relationships and determine the driving criteria of CWBE development. The results showed that
            government strategies play a causal role in CWBE development and drive digital collaboration, smart operations,
            and supply chain integration. The findings also indicated that smart manufacturing technology, organizational
            policies, market enhancement, supply chain analytics, and operational innovation are drivers of waste integration
            from fisheries into the circular economy through waste-based bioeconomy processes.'''
    
    e3_w2r = {"waste": ["fish waste"],
                    "transforming_process": [],
                    "transformed_resource": []}
    
    e4 = '''The maintaining solution for keeping and storing waste over the last century has been landfilling
            as its costs are the lowest. A sustainable approach such as Landfill Mining (LFM) can be applied to recover Rare Earth
            Elements (REEs) and other valuable metals from waste that make fundamental assets in terms of economy and essential
            for developing industrial technologies. This study investigated concentrations of REEs and other metals in waste material.
            Samples from Ida-Virumaa (Estonia) landfilled waste fine fraction was taken to see the element concentration proceeded
            through sequential extraction. Additionally, the method of clay modification was developed that may serve as a sorbent
            to extract the REEs from the inert landfill fine fraction waste using hydroxyapatite modified clay. The amount of REEs
            might become of industrial interest if a feasible landfill mining approach for remediation of landfills and degraded
            industrial soils would be applied together with innovative recovery methods, e.g., sorption by modified clays.'''
    
    e4_w2r = {"waste": ["inert landfill fine fraction waste"],
                    "transforming_process": ["sorption using hydroxyapatite modified clay"],
                    "transformed_resource": ["rare earth elements"]}

    e5 = '''In the pursuit of sustainable energy sources, considerable focus has been directed towards the
            hydrogen production using renewable energy-driven water electrolysis. The exploration of alternative anodic
            reactions, particularly incorporating wastes as electron donors, not only reduces energy consumption of water
            electrolysis but also extends sustainability beyond hydrogen production. This review highlights the significance
            of the chemical properties, widespread availability, and potential value-added prospects of wastes for their
            integration into hybrid water electrolysis systems. Recent advances on the role of anode catalysts in waste
            valorization are critically evaluated, focusing on promising waste streams such as hydrogen sulfide, crude
            glycerol, cellulosic wastes, and plastics. Among them, stability stands out as the primary challenge in the
            rational design of catalysts for sulfide valorization, while achieving enhanced product selectivity is a
            crucial consideration for catalysts involved in the valorization of organic wastes. Furthermore, the development
            of earth-abundant metal catalysts for waste valorization poses a shared challenge, necessitating a profound
            understanding of structure-activity relationship and the fine-tuning of the nanostructure, chemical composition,
            and electronic structure of catalysts.'''
    
    e5_w2r = {"waste": ["hydrogen sulfide", "crude glycerol", "cellulosic wastes", "plastics"],
                    "transforming_process": ["integration into hybrid water electrolysis systems"],
                    "transformed_resource": ["hydrogen"]}
    
    e6 = '''Aluminium smelter waste (ASW) is a big contributor to landfills, and its recycling has been of great
            interest. This study investigates the tribological properties of aluminium matrix syntactic foams manufactured
            using an Al 6082 alloy and ASW. Ball-on-disc tests were conducted under both dry and lubricated conditions.
            Underdry sliding conditions, the coefficient of friction (COF) had an initial sharp increase, followed by a gradual
            decrease and finally a steady state as the sliding distance increased. The wear surfaces showed the presence of
            adhesive, abrasive and oxidative wear, with some presence of delamination. Syntactic foams containing small ASW
            particles led to a decrease in surface roughness, decrease in the average COF and decrease in specific wear.
            Heating large ASW particles before manufacturing the syntactic foams enhanced overall wear properties because
            the particles are hardened due to a compositional change. The T6 treatment of the syntactic foams enhanced the
            wear properties due to the hardening of the Al matrix. The average COF of the ASW syntactic foams was higher than
            that of the E-sphere syntactic foam, which was predominantly abrasive wear. The specific wear of the ASW syntactic
            foams can be higher or lower than the E-sphere syntactic foam, depending on the ASW particle size. Under lubricated
            sliding test conditions, the wear was reduced significantly, and the type changed from predominantly adhesive to
            predominantly abrasive. The porous ASW particles acted as lubricant reservoirs and provided a constant supply of
            lubricant, further improving the lubrication effect.'''
    
    e6_w2r = {"waste": ["aluminium smelter waste"],
                    "transforming_process": ["manufacturing", "heating", "T6 treatment"],
                    "transformed_resource": ["aluminium matrix syntactic foams"]}
    
    e7 = '''Depolymerization of carbohydrate biomass using a long-chain alcohol (transglycosylation) to produce alkyl
            glycoside-based bio-surfactants has been gaining industrial interest. This study introduces microwave-assisted
            transglycosylation in transforming wheat bran, a substantial agricultural side stream, into these valuable compounds.
            Compared to traditional heating, microwave-assisted processing significantly enhances the product yield by 53 % while
            reducing the reaction time by 72 %, achieving a yield of 29 % within 5 h. This enhancement results from the microwave's
            capacity to activate intermolecular hydrogen and glycosidic bonds, thereby facilitating transglycosylation. Life-cycle
            assessment and techno-economic analysis demonstrate the benefits of microwave heating in reducing energy consumption by
            42 %, CO2 emissions by 56 %, and equipment, operational and production costs by 44 %, 35 % and 30 %, respectively. The
            study suggests that microwave heating is a promising approach for efficiently producing bio-surfactants from agricultural
            wastes, with potential cost reductions and environmental benefits that could enhance industrial biomass conversion processes.'''
    
    e7_w2r = {"waste": ["carbohydrate biomass", "wheat bran"],
                    "transforming_process": ["microwave-assisted transglycosylation"],
                    "transformed_resource": ["alkyl glycoside-based bio-surfactants"]}
    
    e8 = '''Conventional management mechanisms for construction waste recycling and reuse (CWRR) often cause considerable information
            asymmetry, inadequate government supervision, and imperfect incentive mechanisms, resulting in frequent illegal dumping and
            landfilling of construction waste and relatively low recycling rates. To address these issues, we introduce blockchain technology
            to CWRR. The design science research methodology was adopted to identify the main steps in functional requirement identification
            and blockchain framework development for CWRR. To explain how blockchain can improve the CWRR process, a conceptual model for the
            functional requirements of the proposed framework was constructed using qualitative analysis. A blockchain-driven framework was
            developed to overcome practical barriers in the CWRR industry. Based on scenario simulation results, the proposed framework had
            execution and transaction costs of $4.735 and $1.276, respectively, and latency performance at the millisecond level. The results
            indicate that (1) based on specific problems systematically identified from CWRR practices, the proposed framework can address
            practical barriers in the CWRR industry more directly; (2) the CWRR industry can use blockchain technology to achieve information
            sharing, comprehensive government supervision, and effective incentive mechanisms and (3) the blockchain-driven framework has high
            efficacy and can promote efficient CWRR and high-quality development of CWRR industry chain. This management model is conducive to
            forming a collaborative CWRR industry chain that can lead to broader adoption of blockchain technology across industries.'''

    e8_w2r = {"waste": ["construction waste"],
                    "transforming_process": [],
                    "transformed_resource": []}
    
    e9 = '''Raspberry seeds are a by-product of berries, both from their primary processing, such as in juice production, and secondary processing,
            such as in oil extraction. These seeds contain plenty of valuable components such as crude fiber, proteins, fats, and vitamins. Quality
            characterization is the initial step toward using these seeds as a sustainable and functional food. The aim of studying raspberry seeds’
            quality profile, both before oil extraction and after different processing methods (supercritical CO2, subcritical CO2, cold pressing,
            and hexane solvent), is to point out the benefits of this by-product and to raise consumer awareness about their health and well-being
            benefits. This study provides evidence that raspberry seeds have good physical parameters for use in other products as a functional
            food enrichment ingredient, such as in baked goods, offering considerable health benefits due to their high nutrient content. The weights,
            peroxide values, moisture content, nutritional energy values, and colors were determined before oil extraction to give initial seed values.
            The nutrient content and amounts of macroelements, P, K, Ca, and Mg, as well as microelements, B, Zn, Cu, Fe, and Mn, were determined in
            the tested variety ‘Polka’, both before and after oil extractions and using different methods. The raspberry seeds’ moisture was 9.2%,
            their peroxide content was 5.64 mEq/kg, their nutritional value was 475.25 Kcal., and their total weight was 2.17 mg (1000 units). The
            seeds contain 7.4% protein, 22.1% crude fiber, 11.0% crude fat and oil, and 2.8% sugar. We determined how different oil extraction methods
            influence the nutrient, micro-, and macro-component values. We concluded that the seeds contained the highest manganese (45.3 mg/kg),
            iron (29.2 mg/kg), and zinc (17.4 mg/kg) contents and the lowest content of copper (5.1 mg/kg). This research shows that raspberry seeds
            represent a potential natural food ingredient, and after oil extraction with subcritical or supercritical CO2 or cold pressing, they can
            be used as a sustainable and functional food.'''
    
    e9_w2r = {"waste": ["raspberry seeds"],
                    "transforming_process": ["oil extraction"],
                    "transformed_resource": ["functional food"]}

    abstract_list = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]
    w2r_list = [e0_w2r, e1_w2r, e2_w2r, e3_w2r, e4_w2r, e5_w2r, e6_w2r, e7_w2r, e8_w2r, e9_w2r]
    return abstract_list, w2r_list
