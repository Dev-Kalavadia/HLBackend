const express = require('express');
const bodyParser = require('body-parser');
import { SentenceTransformer } from '@botisan-ai/sentence-transformers';

const app = express();
const port = 4334;

app.use(bodyParser.json());

// Initialize the sentence-transformer model
let model;
let tagEmbeddings;
const tags = ['environment', 'geographical', 'ocean', 'sea', 'river', 'land', 'mountain', 'glacier',
    'arctic', 'valley', 'oasis', 'forest', 'desert', 'canyon', 'climate', 'temperate', 'tropic',
    'humid', 'arid', 'polar-climate', 'desert-climate', 'natural-disasters', 'cultural-effects',
    'population-density', 'pollution', 'climate-change', 'religious-practices', 'spiritualism',
    'mythology', 'symbolism', 'sacred-cities', 'mecca', 'jerusalem', 'sacred-landscapes-beings',
    'pilgrims-route', 'beaches', 'trees', 'rocks', 'sacred-places', 'church', 'mosque', 'masjid',
    'synagogue', 'temple', 'cemetery', 'columbarium', 'chapel', 'shrine', 'multi-layered-sanctity',
    'hagia-sophia', 'belief-systems', 'islam', 'christianism', 'judaism', 'sufism', 'hinduism', 'buddhism',
    'shamanism', 'nature-worshiping', 'new-age-beliefs', 'religious-music', 'islamic-music', 'christian-music',
    'jewish-music', 'buddhist-music', 'hindu-music', 'shamanic-music', 'new-age-music', 'religious-art', 'religious-conflict',
    'rites-ceremonies', 'economic-activities', 'trade-routes', 'caravansary', 'spice-route', 'silk-road', 'indian-ocean',
    'navigation', 'markets', 'open-bazaars', 'covered-bazaars', 'historical-bazaars', 'bakkalas', 'crossroad-cities',
    'transportation', 'caravansaries', 'cosmopolitanism', 'culture-amalgamation', 'port-cities', 'transportation-dhow',
    'cosmopolitan', 'cultural-amalgamation', 'importation-exportation', 'commercial-activities-institutions', 'colonialism', 'colonialism-companies', 'taxation', 'tourism', 'cultural-tourism', 'mass-tourism', 'business-tourism', 'dark-heritage-tourism', 'skills', 'fishing', 'pearling', 'agriculture', 'mining',
    'water-management', 'shipbuilding', 'artisanship', 'arts-communication', 'performing-arts', 'music', 'dance', 'theatre',
    'opera', 'media-arts', 'installation-art', 'film', 'digital-art', 'visual-arts', 'painting', 'sculpture', 'masks', 'crafts',
    'literary-arts', 'poetry', 'novel', 'fiction', 'cultural-social-communication', 'narratives', 'storytelling', 'oral-history',
    'proverbs', 'rhetoric', 'poetry-cultural', 'quotes', 'literature', 'museum-exhibitions', 'media-presentation', 'travel',
    'rites-ceremonies-communication', 'social-structures-governance', 'social-structures', 'status', 'religion', 'age', 'ethnicity',
    'class', 'gender', 'slavery', 'discrimination', 'family', 'household', 'neighbors', 'community', 'citizenship', 'outsiders',
    'patriarchy', 'matriarchy', 'socio-political-communities', 'governance', 'monarchy', 'democracy', 'dictatorship',
    'tribal-authority', 'sheikhdom', 'shariah', 'secular', 'living-societies', 'disappeared-societies', 'past-societies',
    'human-interaction', 'mobility', 'expedition', 'tourism-mobility', 'education', 'trade', 'migration', 'immigration',
    'displacement', 'colonization', 'exile', 'slavery-mobility', 'cultural-amalgamation-human', 'cosmopolitanism-cultural',
    'multi-culturality', 'human-habitation', 'past-ancient-settlements', 'preliterate-period', 'bronze-age', 'iron-age', 'medieval',
    'religious-buildings', 'administrative-buildings', 'historical-buildings', 'ruins', 'world-heritage-site', 'modern-contemporary-settlements',
    'villages', 'towns', 'cities', 'mega-cities', 'ruins-settlements', 'suburbs', 'ghettos', 'gentrificated-districts',
    'continuous-settlements', 'living-heritage-site', 'world-heritage-site-continuous', 'old-towns-of-the-cities',
    'settlement-elements', 'palaces', 'monuments', 'residential-buildings', 'administrative-buildings-settlement', 'plaza',
    'graveyards', 'neighborhoods', 'ethnic-clusters', 'central-suburb', 'slum-squatted-spaces', 'tradition', 'traditional-knowledge',
    'oral-traditions', 'traditional-training', 'traditional-agriculture', 'traditional-art', 'vernacular-architecture',
    'shipbuilding-traditional', 'traditional-food-beverage', 'winemaking', 'olive-oil-process', 'bread-making',
    'ceremonial-feasts', 'festival-feasts', 'funeral-feasts', 'circumcise-feasts', 'religious-feasts-iftar',
    'treating-neighbor-with-food', 'welcoming-stranger', 'sacrification', 'fruit-offering', 'traditional-healing',
    'devil-dances', 'masks-healing', 'ayurvedic', 'religious-healing', 'traditions-under-threat', 'science-technology',
    'science', 'medicine', 'maps-mapping', 'conservation', 'mineralogy', 'technology', 'pyrotechnology', 'architecture',
    'shipbuilding-technology', 'automobile', 'navigation-technology', 'agriculture-tools', 'hunting-tools',
    'pearl-diving-tools', 'communication-tools', 'shared-contested', 'memory-identity', 'resistance', 'conflict', 'wars',
    'indigenous-peoples', 'identity', 'memory-politics', 'revolution', 'colonization-memory', 'decolonization',
    'reconciliation', 'heritage-preservation', 'heritage-management', 'heritage-tourism', 'heritage-diplomacy',
    'heritage-ownership']






app.use(bodyParser.json());

let similarityModel;

// Asynchronously load the sentence similarity model
async function loadModel() {
    const transformers = await import('@xenova/transformers');
    similarityModel = await transformers.pipeline('sentence-similarity', 'mixedbread-ai/mxbai-embed-large-v1');
}

app.post('/get_related_tags', async (req, res) => {
    const { sentence } = req.body;
    if (!sentence) {
        return res.status(400).json({ error: 'Sentence is required' });
    }

    try {
        const results = await similarityModel(sentence);
        const similarTags = results.map(item => item.label); // Adjust based on actual output
        res.json({ tags: similarTags.slice(0, 3) });
    } catch (error) {
        console.error('Error processing the request:', error);
        res.status(500).send('Error processing the request');
    }
});


app.listen(port, async () => {
    console.log(`Server running on port ${port}`);
    await loadModel();
});
