import xml.etree.ElementTree as ET
import re


def convert_text_to_weighted_xml(input_file_path, output_file_path):
    """
    Converts a given text file into a weighted XML file.

    Args:
    input_file_path (str): The path to the input text file.
    output_file_path (str): The path where the output XML file will be saved.
    """

    def parse_document(file_path):
        with open(file_path, "r") as file:
            return file.readlines()

    def create_xml_structure(lines):
        root = ET.Element("Document")
        current_section = None
        current_subsection = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                section = ET.SubElement(root, "Section", weight="1.0")
                section_title = ET.SubElement(section, "Title")
                section_title.text = line
                current_section = section
                current_subsection = None
            elif "1.1." in line or "2.1." in line:
                subsection = ET.SubElement(current_section, "Subsection", weight="0.8")
                subsection_title = ET.SubElement(subsection, "Title")
                subsection_title.text = line
                current_subsection = subsection
            else:
                parent_element = (
                    current_subsection if current_subsection else current_section
                )
                if not parent_element:
                    parent_element = root
                content = ET.SubElement(parent_element, "Content", weight="0.5")
                content.text = line

        return root

    lines = parse_document(input_file_path)
    xml_root = create_xml_structure(lines)

    tree = ET.ElementTree(xml_root)
    tree.write(output_file_path, encoding="utf-8", xml_declaration=True)


# Example usage
convert_text_to_weighted_xml(
    "python/text_files/example1.txt", "python/text_files/example1.xml"
)
