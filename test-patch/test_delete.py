import xml.etree.ElementTree as ET

# Parse the XML file
tree = ET.parse('scenes/00813-svBbv1Pavdk/00813-svBbv1Pavdk.xml')
root = tree.getroot()

# Loop through each <sensor> element
for sensor in root.findall('.//sensor'):
    # Remove the id and name attributes
    del sensor.attrib['id']
    del sensor.attrib['name']

# Save the modified XML file
tree.write('scenes/00813-svBbv1Pavdk/00813-svBbv1Pavdk.xml')