'use client';

import { useState } from 'react';
import styles from './Claim.module.css';

export default function ClaimComponent({ claim, isSelected, onClick, onTagClick }) {
    const [isClassifying, setIsClassifying] = useState(false);
    const [suggestedCategory, setSuggestedCategory] = useState(null);

    const handleTagClick = (e) => {
        e.stopPropagation();

        if (claim.tag && onTagClick) {
            onTagClick(claim.tag);
        }
    };

    const handleClassify = async (e) => {
        e.stopPropagation();
        setIsClassifying(true);
        setSuggestedCategory(null);

        try {
            const response = await fetch('/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: claim.content }),
            });

            if (!response.ok) {
                throw new Error('Classification failed');
            }

            const data = await response.json();
            setSuggestedCategory(data.category);
        } catch (error) {
            console.error(error);
            setSuggestedCategory('Erreur, réessaie');
        } finally {
            setIsClassifying(false);
        }
    };

    return (
        <div
            className={`${styles.container} ${isSelected ? styles.selected : ''}`}
            onClick={onClick}
            role="button"
            tabIndex={0}
            aria-pressed={isSelected}
        >
            <div className={styles.content}>
                <p
                    className={styles.text}
                    id={`claim-content-${claim.id}`}
                >
                    {claim.content}
                </p>
                {claim.tag && (
                    <button
                        className={styles.tag}
                        onClick={handleTagClick}
                        aria-label={`Navigate to ${claim.tag} inbox`}
                        title={`Go to ${claim.tag} inbox`}
                    >
                        {claim.tag}
                    </button>
                )}

                <button
                    onClick={handleClassify}
                    disabled={isClassifying}
                >
                    {isClassifying ? 'Analyse en cours...' : 'Suggérer une catégorie (IA)'}
                </button>

                {suggestedCategory && (
                    <p>
                        Catégorie suggérée : <strong>{suggestedCategory}</strong>
                    </p>
                )}
            </div>
        </div>
    );
}
